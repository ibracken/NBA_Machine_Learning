import subprocess
from scripts.clusterScraper import run_cluster_scraper
from scripts.boxScoreScraper import run_scrape_box_scores
from scripts.dailyPredictionsScraper import run_daily_predictions_scraper

def run_notebook(notebook_path):
    """
    Executes a Jupyter Notebook file programmatically.
    :param notebook_path: Path to the .ipynb file to execute.
    """
    try:
        print(f"Running notebook: {notebook_path}")
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", notebook_path],
            check=True,
        )
        print(f"Notebook executed successfully: {notebook_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running notebook: {notebook_path}")
        print(e)



def main():
# clusterScraper.py
    print("Starting Cluster Scraper...")
    run_cluster_scraper()
# nbaClustering.ipynb
    print("Executing NBA Clustering Notebook...")
    run_notebook("scripts/nbaClustering.ipynb")
# boxScoreScraper.py
    print("Starting Box Score Scraper...")
    run_scrape_box_scores()
# nbaSupervisedLearningClusters.ipynb
    print("Executing NBA Supervised Learning Clusters Notebook...")
    run_notebook("scripts/nbaSupervisedLearningClusters.ipynb")

# dailyPredictionsScraper.py
    print("Starting Daily Predictions Scraper...")
    run_daily_predictions_scraper()
if __name__ == "__main__":
    main()