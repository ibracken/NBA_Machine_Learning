"""
Thin wrapper over the Anthropic SDK: every call streams, opts into refusal fallbacks, sets effort
explicitly, fails loudly on refusal or truncation, and records token usage and cost.
"""

import json
import logging
import os

import anthropic

from config import FALLBACK_BETA, PAUSE_TURN_LIMIT, PRICING

logger = logging.getLogger()

# One record per API call made during this invocation; lambda_function persists it
USAGE = []

_client = None


class LLMError(RuntimeError):
    pass


def client():
    global _client
    if _client is None:
        # Key comes from the CLAUDE_API_KEY environment variable (Lambda config, or .env for local runs)
        api_key = os.environ.get('CLAUDE_API_KEY')
        if not api_key:
            raise LLMError("CLAUDE_API_KEY is not set")
        _client = anthropic.Anthropic(api_key=api_key, max_retries=4)
    return _client


def _record_usage(label, requested_model, response):
    usage = response.usage
    server_tools = getattr(usage, 'server_tool_use', None)
    record = {
        'label': label,
        'requested_model': requested_model,
        'served_model': response.model,
        'stop_reason': response.stop_reason,
        'input_tokens': usage.input_tokens or 0,
        'output_tokens': usage.output_tokens or 0,
        'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', 0) or 0,
        'cache_read_input_tokens': getattr(usage, 'cache_read_input_tokens', 0) or 0,
        'web_search_requests': (getattr(server_tools, 'web_search_requests', 0) or 0) if server_tools else 0,
    }
    # Web search requests are billed separately and are not included in cost_usd
    rates = PRICING.get(response.model) or PRICING.get(requested_model)
    if rates:
        rate_in, rate_out, rate_write, rate_read = rates
        record['cost_usd'] = round((record['input_tokens'] * rate_in + record['output_tokens'] * rate_out
                                    + record['cache_creation_input_tokens'] * rate_write
                                    + record['cache_read_input_tokens'] * rate_read) / 1e6, 5)
    else:
        record['cost_usd'] = None
        logger.warning(f"{label}: no pricing for {response.model}; cost not recorded")
    if response.model != requested_model:
        logger.warning(f"{label}: served by fallback model {response.model} (requested {requested_model})")
    USAGE.append(record)
    return record


def _check_stop(label, response):
    if response.stop_reason == 'refusal':
        details = getattr(response, 'stop_details', None)
        raise LLMError(f"{label}: refused (category={getattr(details, 'category', None)}, "
                       f"explanation={getattr(details, 'explanation', None)})")
    if response.stop_reason in ('max_tokens', 'model_context_window_exceeded'):
        raise LLMError(f"{label}: output cut off ({response.stop_reason})")


def call(label, model, effort, system, messages, max_tokens, tools=None, output_schema=None, cache=False):
    """One streamed request. Returns the final message; stop_reason is checked by the caller."""
    output_config = {'effort': effort}
    if output_schema is not None:
        output_config['format'] = {'type': 'json_schema', 'schema': output_schema}
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config=output_config,
        betas=[FALLBACK_BETA],
        fallbacks='default',
    )
    if tools:
        kwargs['tools'] = tools
    if cache:
        kwargs['cache_control'] = {'type': 'ephemeral'}

    with client().beta.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    record = _record_usage(label, model, response)
    logger.info(f"{label}: {record['served_model']} stop={response.stop_reason} in={record['input_tokens']} "
                f"out={record['output_tokens']} searches={record['web_search_requests']} ${record['cost_usd']}")
    return response


def research(label, model, effort, system, prompt, tools, max_tokens=32000):
    """
    A web-search call. Returns (briefing_text, citations) where each citation is the exact snippet
    of a search result the text relies on: {'url', 'title', 'cited_text'}.
    """
    messages = [{'role': 'user', 'content': prompt}]
    texts, citations, seen = [], [], set()
    for _ in range(PAUSE_TURN_LIMIT + 1):
        response = call(label, model, effort, system, messages, max_tokens, tools=tools)
        for block in response.content:
            if block.type != 'text':
                continue
            texts.append(block.text)
            for citation in block.citations or []:
                if citation.type != 'web_search_result_location':
                    continue
                key = (citation.url, citation.cited_text)
                if key not in seen:
                    seen.add(key)
                    citations.append({'url': citation.url, 'title': citation.title,
                                      'cited_text': citation.cited_text})
        if response.stop_reason != 'pause_turn':
            _check_stop(label, response)
            return ''.join(texts).strip(), citations
        # The server-side search loop paused; resend so it resumes where it stopped
        messages = [messages[0], {'role': 'assistant', 'content': response.content}]
    raise LLMError(f"{label}: still paused after {PAUSE_TURN_LIMIT} continuations")


def extract(label, model, effort, system, prompt, schema, max_tokens=32000):
    """A no-tools call constrained to a JSON schema. Returns the parsed object."""
    response = call(label, model, effort, system, [{'role': 'user', 'content': prompt}], max_tokens,
                    output_schema=schema)
    _check_stop(label, response)
    text = next((b.text for b in response.content if b.type == 'text'), None)
    if text is None:
        raise LLMError(f"{label}: no text block in structured response")
    return json.loads(text)


def check_stop(label, response):
    _check_stop(label, response)
