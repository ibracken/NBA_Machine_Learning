[CmdletBinding()]
param(
    [string]$Region = "us-east-1",
    [string]$AccountId = $env:AWS_ACCOUNT_ID,
    [string[]]$Functions = @(
        "cluster-scraper",
        "nba-clustering",
        "box-score-scraper",
        "supervised-learning",
        "daily-predictions",
        "injury-scraper",
        "minutes-projection"
    ),
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-CommandPathOrThrow {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Required command '$Name' not found in PATH."
    }
    return $cmd.Source
}

function Get-LambdaPolicyDocument {
    param(
        [string]$AwsCmd,
        [string]$FunctionName,
        [string]$RegionName
    )

    $policyText = & $AwsCmd lambda get-policy `
        --function-name $FunctionName `
        --region $RegionName `
        --query Policy `
        --output text 2>$null

    if (-not $policyText) {
        return $null
    }

    return ($policyText | ConvertFrom-Json)
}

function Get-StatementsToRemove {
    param([object]$PolicyDoc)

    if (-not $PolicyDoc -or -not $PolicyDoc.Statement) {
        return @()
    }

    $remove = @()
    foreach ($stmt in $PolicyDoc.Statement) {
        $sid = [string]$stmt.Sid
        if (-not $sid) { continue }

        $principalService = ""
        if ($stmt.Principal -and $stmt.Principal.Service) {
            $principalService = [string]$stmt.Principal.Service
        }

        $sourceArn = ""
        if ($stmt.Condition -and $stmt.Condition.ArnLike) {
            if ($stmt.Condition.ArnLike."AWS:SourceArn") {
                $sourceArn = [string]$stmt.Condition.ArnLike."AWS:SourceArn"
            } elseif ($stmt.Condition.ArnLike."aws:SourceArn") {
                $sourceArn = [string]$stmt.Condition.ArnLike."aws:SourceArn"
            }
        }
        if (-not $sourceArn -and $stmt.Condition -and $stmt.Condition.ArnEquals) {
            if ($stmt.Condition.ArnEquals."AWS:SourceArn") {
                $sourceArn = [string]$stmt.Condition.ArnEquals."AWS:SourceArn"
            } elseif ($stmt.Condition.ArnEquals."aws:SourceArn") {
                $sourceArn = [string]$stmt.Condition.ArnEquals."aws:SourceArn"
            }
        }

        $isEventBridgeSid = $sid.StartsWith("EventBridge-")
        $isSchedulerRule = $sourceArn.Contains(":rule/nba-slate-") -or $sourceArn.Contains(":rule/nba-game-")
        $isEventBridgePrincipal = $principalService -eq "events.amazonaws.com"

        if ($isEventBridgeSid -or ($isEventBridgePrincipal -and $isSchedulerRule)) {
            $remove += $sid
        }
    }

    return $remove | Select-Object -Unique
}

$aws = Get-CommandPathOrThrow -Name "aws"

if (-not $AccountId) {
    throw "AccountId is required. Pass -AccountId or set AWS_ACCOUNT_ID."
}

$mode = if ($Apply) { "APPLY" } else { "DRY-RUN" }
Write-Host "Mode: $mode"
Write-Host "Region: $Region"
Write-Host "AccountId: $AccountId"
Write-Host ""

$sourceArn = "arn:aws:events:${Region}:${AccountId}:rule/nba-slate-*"
$statementId = "EventBridge-nba-slate"

foreach ($fn in $Functions) {
    Write-Host "=== $fn ==="

    $beforePolicy = Get-LambdaPolicyDocument -AwsCmd $aws -FunctionName $fn -RegionName $Region
    $beforeLen = if ($beforePolicy) { ($beforePolicy | ConvertTo-Json -Depth 20 -Compress).Length } else { 0 }
    Write-Host "Policy size (approx chars): $beforeLen"

    $toRemove = Get-StatementsToRemove -PolicyDoc $beforePolicy
    if ($toRemove.Count -eq 0) {
        Write-Host "No stale EventBridge statements found."
    } else {
        Write-Host ("Will remove {0} statement(s): {1}" -f $toRemove.Count, ($toRemove -join ", "))
    }

    if ($Apply) {
        foreach ($sid in $toRemove) {
            try {
                & $aws lambda remove-permission `
                    --function-name $fn `
                    --statement-id $sid `
                    --region $Region 1>$null
                Write-Host "Removed: $sid"
            } catch {
                Write-Warning "Failed removing $sid on ${fn}: $($_.Exception.Message)"
            }
        }

        try {
            & $aws lambda add-permission `
                --function-name $fn `
                --statement-id $statementId `
                --action lambda:InvokeFunction `
                --principal events.amazonaws.com `
                --source-arn $sourceArn `
                --region $Region 1>$null
            Write-Host "Added/ensured wildcard permission: $statementId"
        } catch {
            $msg = $_.Exception.Message
            if ($msg -like "*ResourceConflictException*") {
                Write-Host "Wildcard permission already exists: $statementId"
            } else {
                throw
            }
        }

        $afterPolicy = Get-LambdaPolicyDocument -AwsCmd $aws -FunctionName $fn -RegionName $Region
        $afterLen = if ($afterPolicy) { ($afterPolicy | ConvertTo-Json -Depth 20 -Compress).Length } else { 0 }
        Write-Host "Policy size after (approx chars): $afterLen"
    }

    Write-Host ""
}

if (-not $Apply) {
    Write-Host "Dry run complete. Re-run with -Apply to execute changes."
}
