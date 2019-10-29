#PBS -l nodes=1:ppn=1,pmem=10gb,pvmem=10gb
#PBS -l walltime=03:00:00
#PBS -N Reddit
#PBS -M dmatthe1+VACC@uvm.edu
#PBS -m bea 
#PBS -j oe
#PBS -o $HOME/job_logs
#PBS -q shortq

# log some information to stdout + cd to the correct directory
echo "This is my job being started on" `hostname`
cd $HOME/OtherProjects

echo `printenv`
echo $RedditMonth

echo ".headers on" >> $RedditMonth.tmp
echo ".mode csv" >> $RedditMonth.tmp
echo "CREATE TABLE IF NOT EXISTS Comments(id10 INT, datetime INT, subreddit TEXT, author TEXT, body TEXT, banned BOOL, suspect BOOL);" >> $RedditMonth.tmp
echo ".import CSV/$RedditMonth.csv Comments" >> $RedditMonth.tmp
echo "CREATE INDEX IF NOT EXISTS datetimeIdx ON Comments (datetime);" >> $RedditMonth.tmp
echo "CREATE INDEX IF NOT EXISTS bannedIdx on Comments(banned);" >> $RedditMonth.tmp
echo "CREATE INDEX IF NOT EXISTS suspectIdx on Comments(suspect);" >> $RedditMonth.tmp


/usr/bin/sqlite3 DBs/$RedditMonth.db < $RedditMonth.tmp

rm  $RedditMonth.tmp
