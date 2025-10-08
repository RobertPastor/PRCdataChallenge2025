## how to download the parquet files

## et the current dir into AnsPRCDataChallenge\trajectory\AdsBtrajectories\AnsPerformanceChallenge

open command lines window -> not a power shell -> only a cmd

Microsoft Windows [version 10.0.22631.4317]
(c) Microsoft Corporation. Tous droits réservés.

C:\Users\rober>

## create access keys

C:\Users\rober\git\PRCdataChallenge2025\Data-Download-OpenSkyNetwork>mc admin accesskey create prc-2025-datasets/
Access Key: ------xxxxx------
Secret Key: ------yyyyy------
Expiration: NONE
Name:
Description:

C:\Users\rober\git\PRCdataChallenge2025\Data-Download-OpenSkyNetwork>

## launch following minio commands

C:\Users\rober\git\PRCdataChallenge2025\Data-Download-OpenSkyNetwork>mc alias set prc-2025-datasets https://s3.opensky-network.org/ ---accesskey---- ---secret key----
Added `prc-2025-datasets` successfully.

C:\Users\rober\git\PRCdataChallenge2025\Data-Download-OpenSkyNetwork>

## download parquet files

mc.exe mirror prc-2025-datasets/  .

this will create a subfolder called competition-data and download the parquet files into