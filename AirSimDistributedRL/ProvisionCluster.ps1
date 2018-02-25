Param(
        [Parameter(Mandatory=$true)]
        [String] $subscriptionId,
        [Parameter(Mandatory=$true)]
        [String] $resourceGroupName,
        [Parameter(Mandatory=$true)]
        [String] $batchAccountName
)

az login
az account set --subscription $subscriptionId 
az batch account set --resource-group $resourceGroupName --name $batchAccountName
az batch pool create --json-file pool.json
