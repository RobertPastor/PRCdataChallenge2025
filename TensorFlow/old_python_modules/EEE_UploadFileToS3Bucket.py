'''
Created on 10 oct. 2025

@author: robert
'''

import os
# file_uploader.py MinIO Python SDK example
from minio import Minio
from minio.error import S3Error

from minio.datatypes import Object
import re

import logging
import unittest


def getLatestTeamSubmittedVersion():

    # create a client
    client = Minio( endpoint = "s3.opensky-network.org" ,
                    access_key = "HertaMoschenPastor" ,
                    secret_key = "HertaMoschenPastor1&&&xxx" ,
                    secure = True)
                    
    
    print("total buckets : " , len ( client.list_buckets() ) )
    for bucket in client.list_buckets():
        print ( bucket.name , bucket.creation_date )
    
    regexp_pattern = r"[.]"
    listOfVersions = []
    for obj in client.list_objects(bucket_name="prc-2025-understated-zucchini", prefix="understated-zucchini"):
        #print ( object.object_name )
        fileName = obj.object_name
        if str(fileName).endswith("parquet"):
            print ( fileName )
            fileVersion = str(fileName.split("_")[1])
            print ( fileVersion )
            fileVersion = re.split(regexp_pattern, fileVersion)
            fileVersion = fileVersion[0]
            print ( fileVersion )
            listOfVersions.append(int(str(fileVersion)[1:]))
            
    listOfVersions.sort()
    print ( listOfVersions)
    return max ( listOfVersions )

#============================================
class Test_Main(unittest.TestCase):

    def test_upload_parquet(self):
        # Create a client with the MinIO server playground, its access key
        # and secret key.
        client = Minio("s3.opensky-network.org",
            access_key="HertaMoschenPastor",
            secret_key="HertaMoschenPastor1&&&xxx",
        )
    
        # The file to upload, change this path if needed
        filesFolder = os.path.dirname(__file__)
        fileName_to_upload  = "understated-zucchini_v1.parquet"
        fileName_to_upload  = "understated-zucchini_v2.parquet"
        fileName_to_upload  = "understated-zucchini_v3.parquet"
        fileName_to_upload  = "understated-zucchini_v4.parquet"
        fileName_to_upload  = "understated-zucchini_v5.parquet"
        fileName_to_upload  = "understated-zucchini_v6.parquet"
        fileName_to_upload  = "understated-zucchini_v7.parquet"
        ''' with outliers '''
        fileName_to_upload  = "understated-zucchini_v8.parquet"
        ''' without outliers replaced by median '''
        fileName_to_upload  = "understated-zucchini_v9.parquet"
        ''' witout outliers replace by capping or clipping to max and min'''
        fileName_to_upload  = "understated-zucchini_v10.parquet"
        
        ''' compute file name to upload '''
        newVersionInt = getLatestTeamSubmittedVersion()+1
        fileName_to_upload = "understated-zucchini_v" + str(newVersionInt) + ".parquet"

        filePath_to_upload = os.path.join(filesFolder , fileName_to_upload)
    
        # The destination bucket and filename on the MinIO server
        bucket_name = "prc-2025-understated-zucchini"
        
        # Make the bucket if it doesn't exist.
        found = client.bucket_exists(bucket_name)
        if found:
            print("Bucket", bucket_name, "already exists")

        # Upload the file, renaming it in the process
        client.fput_object(
            bucket_name, fileName_to_upload, filePath_to_upload,
        )
        print(
            fileName_to_upload, "successfully uploaded as object",
            fileName_to_upload, "to bucket", bucket_name,
        )
        

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        unittest.main()()
    except S3Error as exc:
        print("error occurred.", exc)