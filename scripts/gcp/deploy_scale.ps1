# deploy_scale.ps1 - Copy code + dictionary to the VM and install the env.
# Usage: ./scripts/gcp/deploy_scale.ps1
$ErrorActionPreference = "Stop"
$gcloud = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$VM = "hfps-scale-cpu"; $ZONE = "us-central1-a"; $REMOTE = "/home/yunathanielkang_gmail_com/hfps"

& $gcloud compute ssh $VM --zone=$ZONE --command="mkdir -p $REMOTE"
# code dirs (scale + reused diffusion core); NOT the manuscript/UAI trees
foreach ($d in @("scale","src","diffusion","configs")) {
  & $gcloud compute scp --recurse "$d" "${VM}:${REMOTE}/" --zone=$ZONE
}
& $gcloud compute scp "scripts/gcp/setup_env.sh" "${VM}:${REMOTE}/" --zone=$ZONE
# the column dictionary (rename to ascii on the VM to avoid locale issues)
& $gcloud compute scp "$env:USERPROFILE\Downloads\HIES_2024_mock_변수타입.csv" "${VM}:${REMOTE}/configs/mock_dictionary.csv" --zone=$ZONE
& $gcloud compute ssh $VM --zone=$ZONE --command="cd $REMOTE && bash setup_env.sh"
Write-Host "Deployed to ${VM}:${REMOTE}"