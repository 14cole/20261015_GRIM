param([string]$ValidationPath = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$validationRoot = (Resolve-Path -LiteralPath $ValidationPath).Path
$validationWorkbook = Join-Path $validationRoot 'GRIM_validation_simple.xlsx'
$validationLayouts = Get-Content -LiteralPath (Join-Path $validationRoot '.work/workbook_layout.json') -Raw | ConvertFrom-Json
$validationApp = $null
$validationBook = $null

function Read-ValidationCell($Sheet, [string]$Address) {
    $range = $Sheet.Range($Address)
    try { return $range.Value2 }
    finally { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($range) }
}

function Test-ValidationNumber($Value) {
    $numeric = $Value -is [double] -or $Value -is [single] -or $Value -is [decimal] -or
               $Value -is [byte] -or $Value -is [int16] -or $Value -is [int32] -or $Value -is [int64]
    if (-not $numeric) { return $false }
    # Excel's COM error values are signed integers, not numerical results.
    if ($Value -is [int32] -and $Value -lt -2000000000) { return $false }
    return -not ([double]::IsNaN([double]$Value) -or [double]::IsInfinity([double]$Value))
}

function Test-ValidationClose($Actual, [double]$Expected, [double]$Absolute = 1e-12, [double]$Relative = 1e-9) {
    return (Test-ValidationNumber $Actual) -and
           ([math]::Abs([double]$Actual - $Expected) -le ($Absolute + $Relative * [math]::Abs($Expected)))
}

function Read-ValidationComparison($Sheet, $Check) {
    $expected = Read-ValidationCell $Sheet $Check.expectedCell
    $actual = Read-ValidationCell $Sheet $Check.actualCell
    $errorValue = Read-ValidationCell $Sheet $Check.errorCell
    $reasons = @()
    $tolerance = $null
    $expectedPhaseError = $null
    $phaseError = $null
    $numbers = (Test-ValidationNumber $expected) -and (Test-ValidationNumber $actual)
    if ($numbers) {
        $tolerance = if ($null -ne $Check.tolerance) { [double]$Check.tolerance } else {
            [double]$Check.atol + [double]$Check.rtol * [math]::Abs([double]$expected)
        }
        if ([math]::Abs([double]$actual - [double]$expected) -gt $tolerance) { $reasons += 'Result differs beyond its original tolerance' }
        switch ($Check.errorMode) {
            'phase' {
                $expectedError = 'n/a (angle)'
                $delta = [double]$actual - [double]$expected
                if ($Check.unit -eq 'rad') { $delta *= 180 / [math]::PI }
                # PowerShell remainder preserves the dividend sign; normalize twice to match Excel MOD.
                $expectedPhaseError = [math]::Abs(((($delta + 180) % 360 + 360) % 360) - 180)
            }
            'db' { $expectedError = [math]::Abs([math]::Pow(10, ([double]$actual - [double]$expected) / 10) - 1) }
            default {
                $expectedError = if ([double]$expected -eq 0) { 'n/a (zero)' } else {
                    [math]::Abs([double]$actual - [double]$expected) / [math]::Abs([double]$expected)
                }
            }
        }
    }
    else {
        if ($null -eq $expected -or $null -eq $actual -or
            (Test-ValidationNumber $expected) -or (Test-ValidationNumber $actual) -or
            ([string]$expected -cne [string]$actual)) {
            $reasons += 'Result does not match the expected text or missing-value state'
        }
        $expectedError = if ($null -eq $actual) { 'missing export' } elseif ($null -eq $expected) { 'missing input' } else { 'n/a' }
    }
    if (Test-ValidationNumber $expectedError) {
        if (-not (Test-ValidationClose $errorValue ([double]$expectedError))) { $reasons += 'Percent error does not match the independent calculation' }
    }
    elseif ([string]$errorValue -cne [string]$expectedError) { $reasons += 'Percent-error text does not match the expected condition' }
    if ($Check.phaseErrorCell) {
        $phaseError = Read-ValidationCell $Sheet $Check.phaseErrorCell
        if ($numbers) {
            if (-not (Test-ValidationClose $phaseError ([double]$expectedPhaseError))) { $reasons += 'Wrapped phase error does not match the independent calculation' }
        }
        elseif ([string]$phaseError -cne 'n/a') { $reasons += 'A nonnumeric phase comparison should display n/a' }
    }
    return [pscustomobject]@{
        row = $Check.row; label = $Check.label; expected = $expected; actual = $actual
        tolerance = $tolerance; error = $errorValue; expected_error = $expectedError
        phase_error = $phaseError; expected_phase_error = $expectedPhaseError; reasons = $reasons
    }
}

try {
    $validationApp = New-Object -ComObject Excel.Application
    $validationApp.Visible = $false
    $validationApp.DisplayAlerts = $false
    $validationApp.AutomationSecurity = 3
    $validationApp.AskToUpdateLinks = $false
    $validationBook = $validationApp.Workbooks.Open($validationWorkbook, 0, $false)
    $validationApp.CalculateFullRebuild()
    $validationResults = @()
    foreach ($layout in $validationLayouts) {
        $sheet = $validationBook.Worksheets.Item($layout.title)
        try {
            $failures = @()
            foreach ($check in $layout.checks) {
                $comparison = Read-ValidationComparison $sheet $check
                if ($comparison.reasons.Count -gt 0) { $failures += $comparison }
            }
            $validationResults += [pscustomobject]@{sheet=$layout.title;checks=$layout.checkCount;failures=$failures}
        }
        finally { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($sheet) }
    }

    # Verify that displayed formulas respond to both source and export edits.
    # Restore every changed value before the workbook is saved.
    $validationDynamic = @()
    $statsLayout = $validationLayouts | Where-Object { $_.slug -eq 'stats' }
    $statsCheck = $statsLayout.checks[0]
    $statsSheet = $validationBook.Worksheets.Item($statsLayout.title)
    $inputCell = $statsSheet.Range($statsLayout.inputCells.p000_0)
    $actualCell = $statsSheet.Range($statsLayout.firstActual)
    $oldInput = $inputCell.Value2
    $oldActual = $actualCell.Value2
    $oldExpected = Read-ValidationCell $statsSheet $statsLayout.firstExpected
    $oldError = Read-ValidationCell $statsSheet $statsCheck.errorCell
    try {
        $inputCell.Value2 = [double]$oldInput + 1.0
        $validationApp.CalculateFullRebuild()
        $changedExpected = Read-ValidationCell $statsSheet $statsLayout.firstExpected
        $changedError = Read-ValidationCell $statsSheet $statsCheck.errorCell
        $sourceEditError = [math]::Abs([double]$oldActual - [double]$changedExpected) / [math]::Abs([double]$changedExpected)
        $validationDynamic += [pscustomobject]@{
            test='Changing a Stats input updates the Excel result and percent error'
            passed=($changedExpected -ne $oldExpected -and $changedError -ne $oldError -and (Test-ValidationClose $changedError $sourceEditError))
            before=$oldExpected;after=$changedExpected;error=$changedError
        }
        $inputCell.Value2 = $oldInput
        $validationApp.CalculateFullRebuild()

        $actualCell.ClearContents()
        $validationApp.CalculateFullRebuild()
        $blankError = Read-ValidationCell $statsSheet $statsCheck.errorCell
        $validationDynamic += [pscustomobject]@{
            test='Clearing a GRIM export displays missing export'
            passed=([string]$blankError -ceq 'missing export');status=$blankError
        }

        $actualCell.Value2 = [double]$oldExpected * 1.1
        $validationApp.CalculateFullRebuild()
        $perturbedError = Read-ValidationCell $statsSheet $statsCheck.errorCell
        $validationDynamic += [pscustomobject]@{
            test='A GRIM export 10 percent above the calculation displays 10 percent error'
            passed=(Test-ValidationClose $perturbedError 0.1);error=$perturbedError
        }
    }
    finally {
        $inputCell.Value2 = $oldInput
        $actualCell.Value2 = $oldActual
        $validationApp.CalculateFullRebuild()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($actualCell)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($inputCell)
    }
    $restored = Read-ValidationComparison $statsSheet $statsCheck
    $validationDynamic += [pscustomobject]@{
        test='Restoring the input and export restores the original comparison'
        passed=($restored.reasons.Count -eq 0 -and (Test-ValidationClose $restored.expected ([double]$oldExpected)) -and (Test-ValidationClose $restored.error ([double]$oldError)))
    }
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($statsSheet)

    $validationFormulaErrors = @()
    foreach ($sheet in $validationBook.Worksheets) {
        $used = $sheet.UsedRange
        $values = $used.Value2
        if ($values -is [array]) {
            for ($r=1; $r -le $values.GetLength(0); $r++) {
                for ($c=1; $c -le $values.GetLength(1); $c++) {
                    $v=$values[$r,$c]
                    if (($v -is [int] -and $v -lt -2000000000) -or
                        ($v -is [System.Runtime.InteropServices.ErrorWrapper]) -or
                        ($v -is [string] -and $v -match '^#(REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!|SPILL!|CALC!)$')) {
                        $validationFormulaErrors += [pscustomobject]@{sheet=$sheet.Name;row=$r;column=$c;value=$v}
                    }
                }
            }
        }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($used)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($sheet)
    }
    # Open the saved workbook on a short worked example, with a formula selected.
    $statsSheet = $validationBook.Worksheets.Item($statsLayout.title)
    [void]$statsSheet.Activate()
    $firstFormulaCell = $statsSheet.Range($statsLayout.firstExpected)
    [void]$firstFormulaCell.Select()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($firstFormulaCell)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($statsSheet)
    $validationBook.Save()
    $report = [pscustomobject]@{engine="Microsoft Excel $($validationApp.Version)";recalculated=$true;dynamic_checks=$validationDynamic;formula_errors=$validationFormulaErrors;operations=$validationResults}
    $report | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $validationRoot 'excel_validation_results.json') -Encoding utf8
    $failureCount = ($validationResults | ForEach-Object { $_.failures.Count } | Measure-Object -Sum).Sum
    [pscustomobject]@{operations=$validationResults.Count;comparisons=($validationResults.checks | Measure-Object -Sum).Sum;failed=$failureCount;formula_errors=$validationFormulaErrors.Count;dynamic_checks=$validationDynamic} | ConvertTo-Json -Depth 5
    if ($failureCount -gt 0 -or $validationFormulaErrors.Count -gt 0 -or @($validationDynamic | Where-Object { -not $_.passed }).Count -gt 0) {
        throw 'Excel validation failed. See validation/excel_validation_results.json for details.'
    }
}
finally {
    if ($validationBook) { $validationBook.Close($false); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($validationBook) }
    if ($validationApp) { $validationApp.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($validationApp) }
    [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
