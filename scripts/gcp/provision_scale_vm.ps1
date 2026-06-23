# provision_scale_vm.ps1 - Create the CPU stress-test VM (budget: ~$160, auto-stop).
# Usage: ./scripts/gcp/provision_scale_vm.ps1
$ErrorActionPreference = "Stop"
$gcloud = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$VM   = "hfps-scale-cpu"
$ZONE = "us-central1-a"
$MACHINE = "n2-highmem-32"   # 32 vCPU / 256 GB
$DISK = "500GB"             # 11GB parquet + large CSV outputs + checkpoints

& $gcloud compute instances create $VM `
  --zone=$ZONE `
  --machine-type=$MACHINE `
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud `
  --boot-disk-size=$DISK --boot-disk-type=pd-balanced `
  --metadata=enable-oslogin=FALSE `
  --labels=purpose=hfps-scale,autostop=on

Write-Host "Provisioned $VM ($MACHINE, $DISK) in $ZONE."
Write-Host "STOP when idle:   $gcloud compute instances stop $VM --zone=$ZONE"
Write-Host "DELETE when done: $gcloud compute instances delete $VM --zone=$ZONE"