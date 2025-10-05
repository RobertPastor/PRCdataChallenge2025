## create repo 
## setup current dir to the repo
cd "C:\Users\rober\git\PRCdataChallenge2025"

## create a git repo
git init
-> check the word (master) showing the current branch

## create the virtual environment

$ python -m venv PRCdataChallenge

rober@RobertPastor MINGW64 ~/git/PRCdataChallenge2025 (master)

## check virtual env
$ ls -al PRCdataChallenge
total 5
drwxr-xr-x 1 rober 197609   0 Oct  4 22:27 ./
drwxr-xr-x 1 rober 197609   0 Oct  4 22:27 ../
drwxr-xr-x 1 rober 197609   0 Oct  4 22:27 Include/
drwxr-xr-x 1 rober 197609   0 Oct  4 22:27 Lib/
drwxr-xr-x 1 rober 197609   0 Oct  4 22:27 Scripts/
-rw-r--r-- 1 rober 197609 341 Oct  4 22:27 pyvenv.cfg

## activate the virtual environment

$ source PRCdataChallenge/Scripts/activate
(PRCdataChallenge)
rober@RobertPastor MINGW64 ~/git/PRCdataChallenge2025 (master)

---> word (PRCdataChallenge) appears surrounded by ()
$
## check python version

$ python --version
Python 3.11.3
(PRCdataChallenge)
rober@RobertPastor MINGW64 ~/git/PRCdataChallenge2025 (master)
