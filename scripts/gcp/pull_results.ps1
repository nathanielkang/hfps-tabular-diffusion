# pull_results.ps1 - fetch the ladder summary back (summaries only, not the giant CSVs).
$gcloud = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$VM = "hfps-scale-cpu"; $ZONE = "us-central1-a"; $REMOTE = "/home/yunathanielkang_gmail_com/hfps/results"
New-Item -ItemType Directory -Force -Path results_gcp | Out-Null
& $gcloud compute scp "${VM}:${REMOTE}/scale_ladder.csv" results_gcp/ --zone=$ZONE
& $gcloud compute scp "${VM}:${REMOTE}/scale_ladder.json" results_gcp/ --zone=$ZONE
& $gcloud compute scp "${VM}:${REMOTE}/ladder_run.log" results_gcp/ --zone=$ZONE
Write-Host "pulled ladder summaries to results_gcp/"