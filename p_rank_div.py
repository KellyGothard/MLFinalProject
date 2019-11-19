
############################# Imports #############################
import pandas as pd
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

def make_args():
    description = 'Generalized jobs submitter for PBS on VACC. Tailored to jobs that can be chunked based on datetime.' \
                  ' Scripts to be run MUST have -o output argument. \n Output will be saved in log files with the first 3' \
                  ' characters of args.flexargs and the start date for the job'
    # Specify directory that reddit posts live in in .pbs script
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--inputdir',
                        help='input directory',
                        required=True,
                        type=str)
    # Where your output should be dumped - recommend making a folder for each pbs script 
    # output to keep it clean, as well as a timestamp so you know the order of error logs
    parser.add_argument('-o',
                        '--outdir',
                        help='output directory (will be passed to args.script with -o argument)',
                        required=True,
                        type=str)
    # Get posts starting on this date
    parser.add_argument('-s',
                        '--startdate',
                        help='optional date to constrain the run',
                        required=False,
                        default=None,
                        type=str)
    # Get posts ending on this date (use same date as startdate if you only want one day of posts)
    parser.add_argument('-e',
                        '--enddate',
                        help='optional date to constrain the run',
                        required=False,
                        default=None,
                        type=str)
    # Take a sample of the posts (float as a percentage of the lines read in) - useful for testing
    parser.add_argument('-f',
                        '--fraction',
                        help='use fraction of posts',
                        required=False,
                        default=None,
                        type=str)
    # Control subreddit for comparision - should already be in csv
    parser.add_argument('-c',
                        '--control',
                        help='Control subreddit for comparision - should already be in csv',
                        required=False,
                        default=None,
                        type=str)   
    return parser.parse_args()
    
def valid_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        msg = "Invalid date format in provided input: '{}'.".format(d)
        raise argparse.ArgumentTypeError(msg)

########################## Read in Data ##########################
        
def read_csv(path):
    return pd.read_csv(path)

def rank_divergence(corpus1_df,corpus2_df):
    merged = corpus1_df.merge(corpus2_df, on = 'word')
    return merged
    
def rankdiv_scatter(rank_df,out):
    rank_df['importance'] = ((0.3*rank_df['rank_div'])**2 - (4*rank_df['rank_x'])**2.3)/1000
    lim = max(rank_df['importance']) * -.1
    
    plt.scatter(x = rank_df['importance'], y = rank_df['rank_div'])
    plt.savefig('importance_div.png')
    plt.close()
    plt.scatter(x = rank_df['importance'], y = rank_df['rank_x'])
    plt.savefig('importance_rank.png')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(20,15))
    rank_df['log_rank_div'] = np.log10(rank_df['rank_div'])
    rank_df['log_rank_x'] = np.log10(rank_df['rank_x'])
    ax.scatter(x = rank_df['log_rank_div'] , y = rank_df['log_rank_x'], s = 1)
    
    for i, txt in enumerate(rank_df['word']):
        p = (1.98*(rank_df['log_rank_x'].iloc[i])+(0.5*rank_df['log_rank_div'].iloc[i]))/7.4
        if rank_df['importance'].iloc[i] > lim:
            ax.annotate(txt, (rank_df['log_rank_div'].iloc[i], rank_df['log_rank_x'].iloc[i]),size = ((1/(2.5*rank_df['log_rank_x'].iloc[i])) + (0.2*rank_df['log_rank_div'].iloc[i]))*19, alpha = 0.8)
        else:
            if random.uniform(0,1) > p:
                ax.annotate(txt, (rank_df['log_rank_div'].iloc[i], rank_df['log_rank_x'].iloc[i]),size = ((1/(2.5*rank_df['log_rank_x'].iloc[i])) + (0.2*rank_df['log_rank_div'].iloc[i]))*15,alpha = 0.8)
            else:
                if random.uniform(0,1) > 0.95:
                    ax.annotate(txt, (rank_df['log_rank_div'].iloc[i], rank_df['log_rank_x'].iloc[i]),size = ((1/(2.5*rank_df['log_rank_x'].iloc[i])) + (0.2*rank_df['log_rank_div'].iloc[i]))*15,alpha = 0.8)

    plt.xlabel('rank_div',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('rank',fontsize=16)
    plt.xlim(0,5)
    plt.gca().invert_yaxis()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(out+'rankdiv_scatter.png')
    plt.close()
    
    

def main():
    args = make_args()
    path = args.inputdir`
    out = args.outdir
    
    rank_df = read_csv(path)
    rank_df = rank_df[rank_df['rank_div'] > 0]
    rankdiv_scatter(rank_df,out)
     
    
if __name__=="__main__":
    main()
