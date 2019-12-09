import matplotlib.pyplot as plt
import numpy as np
import random


def rankdiv_scatter(rank_df,col,name):
    rank_df['log_rank_div'] = np.log10(rank_df[col] + 1)
    rank_df['log_rank_x'] = np.log10(rank_df['rank_x'] + 1)
    
    fig, ax = plt.subplots(figsize=(12,16))
    ax.scatter(x = rank_df['log_rank_div'] , y = rank_df['log_rank_x'], s = 1)
    
    for i, txt in enumerate(rank_df['word']):
        if random.uniform(0,1) < 0.05:
            if col == 'rank_div':
                ax.annotate(txt, (rank_df['log_rank_div'].iloc[i], rank_df['log_rank_x'].iloc[i]),size = 1000*abs(rank_df['log_rank_div'].iloc[i]), alpha = 0.5)
            elif col == 'rank_div_crude':
                ax.annotate(txt, (rank_df['log_rank_div'].iloc[i], rank_df['log_rank_x'].iloc[i]),size = 4*rank_df['log_rank_div'].iloc[i], alpha = 0.8)
            
    plt.xlabel('rank_div',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('rank',fontsize=16)
    plt.gca().invert_yaxis()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('../images/'+name.lower()+'_rankdiv_scatter.png')
    plt.show()
    plt.close()


def rankdiv_shift(rank_df,name):
    
    rank_df = rank_df.sort_values('rank_div_crude',ascending=False)
    
    min_index = 0
    max_index = 15
    rank_df = rank_df[min_index:max_index]

    plt.figure(figsize=(6, 8))
    ax = plt.barh(np.arange(len(rank_df)), rank_df['rank_div_crude'], color = 'white', edgecolor='black')
    rects = ax.patches
  
    for rect in rects:
        y_value = rect.get_y() + rect.get_height() / 2
    
        space = 2
        ha = 'left'
        top_rankdiv_words = 0
        if list(rank_df['rank_div_crude'])[rects.index(rect)] > top_rankdiv_words:
            label = str(list(rank_df['word'])[rects.index(rect)]) + ', ' + str(list(rank_df['rank_x'])[rects.index(rect)]) + ', ' + str(list(rank_df['rank_div_crude'])[rects.index(rect)])+ ', ' + str(list(rank_df['count_x'])[rects.index(rect)])
    
            plt.annotate(
                label,                      
                (0, y_value),         
                xytext=(space, 0),          
                textcoords="offset points", 
                va='center',                
                ha=ha)
            
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel('Divergence')
    plt.xticks()
    plt.gca().invert_yaxis()
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.xlim(0,max(rank_df['rank_div_crude'])+100)
    plt.title('word, rank, rank div, count',loc='left',pad=0)
    plt.savefig("../images/"+name+"_rankdiv_shift.png")
    plt.close()
    
    
def rank_divergence_dist_plot(rank_df, name):
    plt.title('Distribution of Rank Divergence for Words in Banned and Not-Banned '+name)
    plt.xlabel('Rank Divergence')
    plt.ylabel('Count')
    plt.hist(rank_df['rank_div_crude'], bins = len(rank_df)//2)
    plt.savefig('../images/'+name.lower()+'_rankdivdistribution.png')
    plt.close()
    
def words_per_row(df,name):
    words_per_row = []
    for index, row in df.iterrows():
        try:
            words = row['body'].split(' ')
        except:
            words = row['comments'].split(' ')
        n_words = len(words)
        words_per_row.append(n_words)
    plt.hist(words_per_row,bins = len(df)//2)
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.title('Words per '+name)
    plt.savefig('../images/'+name.lower()+'_word_dist.png')