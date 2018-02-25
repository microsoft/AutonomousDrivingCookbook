Param(
        [Parameter(Mandatory=$true)]
        [String] $subscriptionId,
        [Parameter(Mandatory=$true)]
        [String] $storageAccountName,
        [Parameter(Mandatory=$true)]
        [String] $storageAccountKey, 
        [Parameter(Mandatory=$true)]
        [String] $resourceGroupName
)

Login-AzureRMAccount
Select-AzureRmSubscription -SubscriptionId $subscriptionId

$cmd = 'azcopy /Source:https://airsimimage.blob.core.windows.net/airsimimage/AirsimImage.vhd /Dest:https://{0}.blob.core.windows.net/prereq/AirsimImage.vhd /destKey:{1}' -f $storageAccountName, $storageAccountKey

write-host $cmd
iex $cmd

$newBlobPath = 'https://{0}.blob.core.windows.net/prereq/AirsimImage.vhd' -f $storageAccountName

$imageConfig = New-AzureRmImageConfig -Location 'EastUs'
$imageConfig = Set-AzureRmImageOsDisk -Image $imageConfig -OsType Windows -OsState Generalized -BlobUri $newBlobPath
$image = New-AzureRmImage -ImageName 'AirsimImage' -ResourceGroupName $resourceGroupName -Image $imageConfig
