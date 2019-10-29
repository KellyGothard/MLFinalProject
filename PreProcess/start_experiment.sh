#PBS -l nodes=1:ppn=1,pmem=1792mb,pvmem=1792mb
#PBS -l walltime=30:00:00
##PBS -t 1-12
#PBS -N Reddit
#PBS -M dmatthe1+VACC@uvm.edu
#PBS -m bea 
#PBS -j oe
#PBS -o $HOME/job_logs
##PBS -q shortq

# log some information to stdout + cd to the correct directory
echo "This is my job being started on" `hostname`
cd $HOME/OtherProjects

python CreateCSV.py $name

echo "done"
