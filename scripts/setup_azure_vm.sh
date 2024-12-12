#!/bin/bash

# Variables
RESOURCE_GROUP="llm-search-rg"
LOCATION="eastus"
VM_NAME="llm-search-vm"
VM_SIZE="Standard_D2s_v3"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create VM
az vm create \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --image UbuntuLTS \
    --admin-username azureuser \
    --generate-ssh-keys \
    --size $VM_SIZE \
    --public-ip-sku Standard

# Open ports
az vm open-port \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --port 80,443

# Get public IP
PUBLIC_IP=$(az vm show \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --show-details \
    --query publicIps \
    --output tsv)

echo "VM created with IP: $PUBLIC_IP" 