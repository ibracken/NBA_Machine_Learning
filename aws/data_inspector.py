import pandas as pd
from s3_utils import load_dataframe_from_s3

def inspect_s3_data():
    """Show what data we have in S3"""
    print("\nInspecting NBA Stats Data...")
    
    try:
        # Load advanced stats
        stats_df = load_dataframe_from_s3('data/advanced_player_stats/current.parquet')
        if not stats_df.empty:
            print("\nAdvanced Stats:")
            print(f"- {len(stats_df)} players, {len(stats_df.columns)} stats")
            print("\nSample Stats:")
            display_cols = ['PLAYER', 'TEAM', 'MIN', 'OFFRTG', 'DEFRTG']
            print(stats_df[display_cols].head().to_string())
            print(f"Last Updated: {stats_df['SCRAPED_DATE'].max()}")
        
        # Load box score data
        print("\nLoading Box Score Data...")
        box_df = load_dataframe_from_s3('data/box_scores/current.parquet')
        if not box_df.empty:
            print("\nBox Score Stats:")
            print(f"- Total Games: {len(box_df)}")
            print(f"- Unique Players: {box_df['PLAYER'].nunique()}")
            print(f"- Date Range: {box_df['GAME_DATE'].min()} to {box_df['GAME_DATE'].max()}")
            
            print("\nSample Box Scores:")
            display_cols = ['PLAYER', 'TEAM', 'GAME_DATE', 'MIN', 'PTS', 'FP', 'CLUSTER']
            print(box_df[display_cols].head().to_string())
            
            # Show some cluster distribution in box scores
            if 'CLUSTER' in box_df.columns:
                print("\nCluster Distribution in Box Scores:")
                cluster_counts = box_df['CLUSTER'].value_counts().sort_index()
                for cluster, count in cluster_counts.items():
                    print(f"Cluster {cluster}: {count} games")
        
        # Load cluster data
        cluster_df = load_dataframe_from_s3('data/clustered_players/current.parquet')
        if not cluster_df.empty:
            print("\nCluster Assignments:")
            print(f"- {len(cluster_df)} players in {cluster_df['CLUSTER'].nunique()} clusters")
            
            # Show distribution of clusters
            cluster_counts = cluster_df['CLUSTER'].value_counts().sort_index()
            print("\nPlayers per Cluster:")
            for cluster, count in cluster_counts.items():
                print(f"Cluster {cluster}: {count} players")
            
            # Sample of players from each cluster
            print("\nSample Players by Cluster:")
            for cluster in sorted(cluster_df['CLUSTER'].unique()):
                sample = cluster_df[cluster_df['CLUSTER'] == cluster]['PLAYER'].head(3).tolist()
                print(f"Cluster {cluster}: {', '.join(sample)}")
            
            print(f"\nLast Clustered: {pd.to_datetime(cluster_df['TIMESTAMP'].max()).strftime('%Y-%m-%d %H:%M:%S')}")
            
    except Exception as e:
        print(f"Error reading S3 data: {str(e)}")

if __name__ == "__main__":
    inspect_s3_data()