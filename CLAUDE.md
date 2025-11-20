This is a lambda environment, so all files within the lambda directory are going to eventually run on lambda.

NEVER MAKE UP INFORMATION

FOR BUILDING MINUTES PREDICTIONS:
* To be able to predict minutes, we have to understand that there are 48 available minutes for each position, so 240 total minutes for a team. There are basically three main factors that change a team’s rotation throughout the course of the season: injuries, role changes, and blowouts.
* Scrape injuries via nba site
* minutes spikes can occurr due to blowouts - predict based on spread
Potential baseline formula:(Average minutes per game for the season * 0.75) + (Average minutes over the last five games * 0.25)