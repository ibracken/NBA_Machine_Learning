import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
from pathlib import Path
import re
from scripts.pdf_text_parser import parse_nba_injury_report

def get_latest_injury_report_url():
    """
    Scrapes the NBA official injury report page to find the most recent PDF link.
    Returns the URL of the latest report.
    """
    page_url = "https://official.nba.com/nba-injury-report-2025-26-season/"

    try:
        print(f"Fetching injury report page: {page_url}")
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all PDF links
        pdf_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'Injury-Report_' in href and href.endswith('.pdf'):
                pdf_links.append(href)

        if not pdf_links:
            raise Exception("No injury report PDFs found on the page")

        # The links are in chronological order, so the last one is the most recent
        latest_url = pdf_links[-1]

        # Extract the time from the URL for display
        time_match = re.search(r'_(\d{2}[AP]M)\.pdf', latest_url)
        time_str = time_match.group(1) if time_match else "unknown time"

        print(f"Found latest report: {time_str}")
        return latest_url

    except Exception as e:
        print(f"Error fetching injury report page: {e}")
        raise


def download_pdf(url, output_dir="/tmp"):
    """
    Downloads a PDF from the given URL to the specified directory.
    Uses /tmp by default for Lambda compatibility.

    Args:
        url: The PDF URL to download
        output_dir: Directory to save the PDF (default: /tmp for Lambda)

    Returns:
        Path to the downloaded PDF file
    """
    try:
        print(f"Downloading PDF from: {url}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract filename from URL
        filename = url.split('/')[-1]
        output_path = os.path.join(output_dir, filename)

        # Download the PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to file
        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"PDF downloaded successfully to: {output_path}")
        print(f"File size: {len(response.content) / 1024:.2f} KB")

        return output_path

    except Exception as e:
        print(f"Error downloading PDF: {e}")
        raise


def fetch_and_parse_latest_injuries(output_dir=None):
    """
    Main function that:
    1. Fetches the latest injury report URL from NBA.com
    2. Downloads the PDF
    3. Parses it
    4. Returns a DataFrame of injuries

    Args:
        output_dir: Directory to save PDF (default: /tmp for Lambda, or ./scripts locally)

    Returns:
        DataFrame with parsed injury data
    """
    # Determine output directory
    if output_dir is None:
        # Use /tmp if running in Lambda, otherwise use ./scripts
        output_dir = "/tmp" if os.environ.get('AWS_LAMBDA_FUNCTION_NAME') else "scripts"

    print("=" * 60)
    print("FETCHING AND PARSING NBA INJURY REPORT")
    print("=" * 60)

    # Get the latest report URL
    pdf_url = get_latest_injury_report_url()

    # Download the PDF
    pdf_path = download_pdf(pdf_url, output_dir)

    # Parse the PDF
    print("\nParsing PDF...")
    df = parse_nba_injury_report(pdf_path)

    print(f"\nSuccessfully parsed {len(df)} injury records (Out status only, excluding G League)")

    return df


if __name__ == "__main__":
    # Run locally for testing
    try:
        df_injuries = fetch_and_parse_latest_injuries(output_dir="scripts")

        print("\nAll injury records:")
        print(df_injuries.to_string())

        # Save to CSV
        output_csv = "parsed_injuries.csv"
        df_injuries.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved to {output_csv}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
