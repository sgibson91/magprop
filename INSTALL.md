# Installing on an Azure Virtual Machine

This document will walk you through setting up a Linux Data Science Virtual Machine on Azure for larger MCMC simulations.
We choose a Data Science Virtual Machine as it comes pre-installed with a [wide range of languages and packages relevant to data science](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/overview#whats-included-in-the-data-science-vm), including Python.

- [Subscription](#Subscription)
- [Creating a Data Science Virtual Machine](#Creating-a-Data-Science-Virtual-Machine)
- [Setting up the Environment](#Setting-up-the-Environment)
- [Running the Code](#Running-the-Code)
  - [Tip: Using `screen`](#Tip-Using-screen)

## Subscription

You will need a subscription to Microsoft Azure in order to follow these instructions.
You can obtain a [free trial subscription](https://azure.microsoft.com/en-gb/free/) which allocates your account $150 for 30 days.
You will be asked to provide a credit card for indentification purposes only.
**You will not be charged.**
At the end of the free trial, your services will automatically cease, then deleted after a given time period if you do not upgrade your account.

Once you have set up your account, we can request a virtual machine.

## Creating a Data Science Virtual Machine

You can also follow [Azure's creation instructions](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro#create-your-data-science-virtual-machine-for-linux).

1. Login into the [Azure Portal](https://portal.azure.com/)
2. From the left panel, click "+ Create a resource"
3. In the search bar, search for "Data Science Virtual Machine for Linux (Ubuntu)", select and click "Create"
4. Fill in the form with the required values:

   * Subscription should be your free trial subscription (unless you have another)
   * For "Resource group", click "Create new" and choose something meaningful, for example "grb-mcmc"
   * Virtual Machine name again should be something meaningful, for example "ds-vm"
   * Region can be anything appropriate for where you are. In the UK, I usually choose "West Europe".
   * For "Size", click "Change size", search for "A8_v2" and select this
   * Under "Administrator account", choose "Password" and assign yourself a username and password
   * Click "Review + create", then "Create" on the next page

This will deploy a Data Science Virtual Machine to your resource group within your subscription.

## Setting up the Environment

The following commands can be run from your local terminal.

1. SSH into your Virtual Machine

```
ssh <USERNAME>@<VM-IP-ADDRESS>
```

To find the IP address of your virtual machine on the Azure Portal:
  * From the left hand panel, select "Resource groups"
  * Select the resource group you created in the last section
  * From the list of resources, click the virtual machine type. It will have the name you assigned as "Virtual machine name" in the last section
  * In the top half of the panel, there will be a field called "Public IP address"

`<USERNAME>` will be the username you assigned yourself in the last section, and you will be prompted for the password you assigned yourself as well.

2. Clone the repository onto the virtual machine

```
git clone https://github.com/sgibson91/magprop.git
cd magprop
```

3. Install the dependencies

```
pip3 install -r requirements
```

## Running the Code

1. Clean the data

```
python3 code/clean_data.py
```

2. Perform a _k_-correction

```
python3 code/kcorr.py --type S
```

3. Run an MCMC simulation

The reason we chose an `A8_v2` size virtual machine in [Creating a Data Science Virtual Machine](#Creating-a-Data-Science-Virtual-Machine) is that it has 8 vCPUs (virtual CPUs).
This means we can run more parallel threads than on a standard desktop or laptop.
We can set the number of threads via the `--n-threads [-d]` argument.

To run a burn-in MCMC simulation on SGRB 050724 with 6 fitting parameters, 200 walkers and 50,000 MC steps:
```
python3 code/mcmc.py \
    --type S \
    --grb 050724 \
    --label <ID-TAG> \
    --n-pars 6 \
    --n-walk 200 \
    --n-step 50000 \
    --burn \
    --n-threads 8
```

where `<ID-TAG>` is a meaningful string so you can identify which run your results have come from.

### Tip: Using `screen`

When executing a long-running command such as this, I like to use `screen`.
This detaches the interactive shell from the physical terminal, allowing your SSH connection to be severed/terminal window to be closed without interrupting or killing the process.
[Read more on `screen` here](https://linux.die.net/man/1/screen).
