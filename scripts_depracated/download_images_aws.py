import boto3
import os

# initial script courtesy of Elrond Cheung me@elrondcheung.com
# %%%
'''
Change the config before you run this script.
This script is used for downloading images which are located in a folder. These images can also be located under multiple levels of (sub)-folders, e.g., 'folder1/subfolder2/'.
In other words, you need to provide the S3 folder path instead of a single file path.

If this script throws an error, please do the followings:
(a) set "verbose" to True AND
(b) send me (i) the create-folder and download-file messages and (ii) the error message.
'''

# !! Important !!
# Change the bucket name to your own bucket name
MY_BUCKET = 'weedimaging'
# Change the directory name to your own directory name. You NEED to change this to the dataset that you want to download.
'''
To download the data from a particular folder, you will need to break the folder path into a list of string.

Example:

'01_Data_capture/Yellangelo/Serrated Tussock/July 20 2022/July 20 2022/'

becomes

['01_Data_capture', 'Yellangelo', 'Serrated Tussock', 'July 20 2022', 'July 20 2022']

'''
# MY_S3_PATH = ['03_Tagged', '2021-10-13']
MY_S3_PATH = ['03_Tagged', '2022-10-13','Serrated Tussock']
# MY_S3_PATH = ['01_Data_capture', 'Yellangelo',
            #   'Serrated Tussock', 'July 20 2022', 'July 20 2022']

# Change this to your AWS MFA ARN
MY_MFA_ARN = 'arn:aws:iam::105587712341:mfa/Dorian.Tsai'

# Prompt user for MFA token (open your MFA app on mobile and input the code in the terminal)
MY_MFA_TOTP = input("Enter the MFA code: ")

# Choose whether to print a message when creating a folder and downloading file
verbose = True

# %%%
'''
Download files from S3
'''

# Create a STS Client
sts_client = boto3.client('sts')
# Create a STS Session with MFA enabled
mfa_creds = sts_client.get_session_token(
    DurationSeconds=36000,
    SerialNumber=MY_MFA_ARN,
    TokenCode=MY_MFA_TOTP,
)
# Create a S3 Client with the STS Session
s3_client = boto3.resource('s3',
                           region_name='ap-southeast-2',
                           aws_access_key_id=mfa_creds['Credentials']['AccessKeyId'],
                           aws_secret_access_key=mfa_creds['Credentials']['SecretAccessKey'],
                           aws_session_token=mfa_creds['Credentials']['SessionToken']
                           )
# Get the bucket
bucket = s3_client.Bucket(MY_BUCKET)

# Create the folder on the local machine if it doesn't exist
local_path = os.path.join(
    os.path.dirname(__file__), os.path.join(*MY_S3_PATH))
if verbose:
    print('> Creating folder: {}'.format(local_path))
os.makedirs(local_path, exist_ok=True)

# Go through the objects in the bucket and folders
for object in bucket.objects.filter(Prefix='/'.join(MY_S3_PATH)):
    if verbose:
        print('> object key: {}'.format(object.key))
    # Create subfolder(s) if necessary
    local_path = os.path.join(
        os.path.dirname(__file__), *object.key.split('/')[:-1])
    if not os.path.exists(local_path):
        if verbose:
            print('> Creating folder: {}'.format(local_path))
        os.makedirs(local_path, exist_ok=True)

    # Download the files
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.download_file
    file = os.path.join(local_path, object.key.split('/')[-1])

    # Do not try to download a path
    if not os.path.isdir(file):
        if verbose:
            print('> Downloading file to: {}'.format(file))
        bucket.download_file(object.key, file)

