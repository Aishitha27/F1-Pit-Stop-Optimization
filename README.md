# PIT STOP STRATEGY OPTIMIZATION IN FORUMLA 1

In Formula 1 motorsport racing, a pit stop refers to a scheduled stop during the race where a driver pulls into the team’s pit box for services such as tire changes, minor repairs, or front wing adjustments. While necessary, pit stops cause a temporary loss of track position as the car slows down, stops, and then rejoins the race. Therefore, the timing of a  pit stop can have a critical impact on a driver's race outcome. A well-timed pit stop not only minimizes time loss but can also create strategic opportunities for overtaking and gaining positions on track. On circuits like Monaco — where overtaking is notoriously difficult due to the narrow, twisting street layout — optimizing pit stop timing becomes even more crucial. Track position is often determined more by pit stop strategies than by raw on-track pace, making race strategy a key battleground for gaining or defending positions. 

The motivation for this project stems from the significant strategic impact of a precisely coordinated and effectively managed pit stop: even a small advantage gained during the pit window can be the difference between achieving a strong finish or becoming trapped behind slower competitors, severely limiting a driver’s race potential.

To address this challenge, this project uses machine learning models trained on historical race data from the Monaco Grand Prix (2018–2024) for drivers from Ferrari, Mercedes, Red Bull, and McLaren — four of the top-performing teams in Formula 1. The provlem this project addresses is twofold:

1. ***Optimal Pit Lap Prediction:*** Estimating the most advantageous lap for executing a pit stop based on tire wear patterns, stint dynamics, and race conditions.

2. ***Short-Term Position Gain Prediction:*** Classifying whether a pit stop at a given lap will result in a net positional gain within a 5-lap window after the stop.

By leveraging features such as tire degradation, stint progression, compound type, lap pace trends, and safety car periods, the models aim to provide data-driven support for pit stop decisions. Focusing on immediate post-pit track position changes enables teams to maximize clear-air opportunities, avoid traffic, and gain critical race advantages — an especially valuable strategy at a circuit like Monaco, where even a single position gain can significantly impact the final outcome.

### Dataset

The dataset for this project was constructed using FastF1 API, a Python library that provides access to official Formula 1 timing and session data. FastF1 allows extraction of structured race information such as lap times, tire compounds, stint details, pit stops, and race positions, enabling detailed race analysis without needing full telemetry (such as throttle, brake, or steering data).

This project focuses on data just from the Monaco Grand Prix across six seasons (2018 to 2024), specifically for drivers from four teams: Ferrari, Mercedes, Red Bull, and McLaren. The data includes one record per completed lap by each driver, summarizing critical race attributes relevant to pit stop strategy.

**Key Features Include:**

1. Year: The race year (2018–2024).

2. Team: Team name (Ferrari, Mercedes, Red Bull, McLaren).

3. Driver: Driver Name(Short Form).

4. LapNumber: Sequential lap number during the race.

5. Stint: Stint number for the driver (a period between two pit stops).

6. Compound: Tire compound used on that lap (e.g., HYPERSOFT, SOFT, MEDIUM).

7. TyreLife: Number of laps completed on the current tire set.

8. StintLengthSoFar: Number of laps completed in the current stint.

9. LapTime(s): Lap time in seconds.

10. LapTimeDelta: Difference in lap time compared to the previous lap.

11. AvgLapTimeInStint: Average lap time during the current stint so far.

12. CumulativeTimeInStint: Total time spent in the current stint.

13. StintDegradationSlope: Slope of lap time degradation (pace loss) across the stint.

14. Position: Driver’s track position at the end of the lap.

15. PositionDelta: Change in position compared to the previous lap.

16. TrackTemp, AirTemp, Humidity, Rain: Weather and track condition indicators.

17. IsSafetyCarLap, IsVSCLap, IsNeutralizedLap: Flags indicating safety car (SC) or virtual safety car (VSC) conditions.

18. CompoundSimplified: Tire compound simplified to Soft, Medium, or Hard categories according to modern regulations.

19. TireEra: Whether the tire regulations were "PreUnified" or "Unified" (reflecting the 2018 tire rule changes).

Starting in 2019, Formula 1 introduced new tire regulations, reducing the number of available tire compounds from seven to five standardized options. To ensure that the machine learning models can accurately differentiate between tire types across different regulation eras, two additional columns were created: CompoundSimplified and TireEra.

*  **CompoundSimplified** maps the original compounds used prior to 2019 to their closest modern equivalents (Soft, Medium, or Hard).
*   **TireEra** identifies whether each lap belongs to the PreUnified (2018) or Unified (2019–2024) tire regulation era.

These adjustments allow the models to consistently interpret tire performance and degradation patterns across seasons with different tire rule frameworks.

### Exploratory Data Analysis

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/71bfae34-6e86-48a1-b65a-4d7f9a9296df" />

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/e5097c72-a5c6-4041-8487-f751817fadc9" />

In earlier years like 2019 and 2020, there are more dips and recoveries in throttle application, especially around the swimming pool chicane and Rascasse. This could suggest car balance issues or conservative exits (rear-end instability).

Verstappen’s throttle control gives him the edge to:

1.   Extend the first stint longer than rivals
2.   Stay out until the pit window is clear (very useful at Monaco, where undercut is tough due to traffic)

2023 was about total control since it started raining mid race — Max was never aggressive, always precise. The crisp braking means less front tire load and better entry 
precision → **confidence under braking, no lockups, which allowed to win the race.**

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/4b45db0f-a2a6-4209-b418-9e503be2b9c0" />

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/81490ff3-9ac9-4036-b87d-d7ab0391407e" />

Brake and throttle smoothness improve noticeably in 2022–2024, signaling Ferrari’s traction and stability improvements. 2024 stands out with one of the cleanest throttle and brake traces, possibly reflecting:

*   Leclerc's continued driver maturity
*   A more refined Ferrari package (power unit + chassis)
*   Optimized energy recovery systems (ERS) allowing better deployment during acceleration zones

But this is likely his best combination of grip, balance, and ERS deployment which led to him finally win the Monaco GP. Leclerc shows improving throttle discipline, especially from 2022–2024 → **key to tire preservation**. This helps extend the first stint — giving Ferrari options to time the stop optimally around Safety Cars or traffic gaps.

<img width="686" height="470" alt="image" src="https://github.com/user-attachments/assets/fe309ff2-bfc7-4c52-a93e-4dc7916467eb" />

Neutralized laps (laps under Safety Car or Virtual Safety Car) vary significantly across different years at Monaco.

* 2019 and 2023 had the highest number of neutralized laps, with 40 and 48 laps respectively — indicating chaotic races with frequent incidents requiring race neutralization.

* 2018 and 2024 show very few neutralized laps (only 7 laps each), suggesting relatively clean races with minimal disruptions.

* 2021 and 2022 had a moderate number of neutralized laps (~25–32 laps), implying some interruptions but not to an extreme degree.

Overall, the number of neutralized laps can strongly influence race strategy — races with many neutralized laps such as Monaco Grand Prix reduce tire degradation and fuel consumption, impacting pit stop timing and stint lengths.

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/a5bcb39d-71ca-43b4-ac27-fe1b634e9128" />

* Soft tyres are mainly used for short stints, typically lasting 5 to 20 laps, prioritizing peak performance but wearing out quickly.

* Medium tyres show balanced usage, commonly lasting 10 to 40 laps, offering flexibility between aggressive and conservative strategies.

* Hard tyres provide the longest lifespan, often lasting 30 to 50+ laps, ideal for minimizing pit stops and executing longer stints.

* Wet and Intermediate tyres are used situationally during changing weather, usually lasting 5 to 20 laps before conditions change.

Majority of tyre stints occur between 5 and 30 laps, highlighting the importance of tyre management in race strategy. 

The stacked distribution illustrates how different compounds dominate at different stages of tyre life: Soft early, Medium mid-range, Hard for endurance.

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/525a6469-7ea8-41ff-8e1c-e88b5f235775" />

LapTimeDelta measures how lap times change compared to the previous lap — negative values mean improvement (faster lap), positive values mean deterioration (slower lap).

In the first few laps (lap 1–3) of a stint:

* Most compounds show initial time loss (slower laps) as tyres heat up and reach optimal grip levels.

* Wet tyres show the most dramatic drop — a major lap time loss (up to -30 seconds) early in the stint, reflecting unstable conditions during rain.

After lap 3:

*   All compounds stabilize, and LapTimeDelta gradually moves closer to zero or positive, meaning lap times become more consistent or slightly slower as tyres begin degrading.

Intermediate tyres recover very quickly after the initial laps, even faster than Dry compounds, indicating better performance as track conditions stabilize.

Hard and Medium tyres show slight positive trends after lap 4–5, suggesting they warm up slower but stay stable longer.

Soft tyres initially improve but plateau quickly, showing limited time gain after initial warm-up.

<img width="851" height="557" alt="image" src="https://github.com/user-attachments/assets/69e13c1a-38c0-46f3-bbbf-f36bda49fe7f" />

There is a clear inverse relationship: as air temperature increases, humidity decreases — this is expected, since warmer air holds more moisture but relative humidity drops.

Soft and Hard tyres are mostly used in lower humidity and higher temperature conditions (around 24–27°C, 40–55% humidity).

Medium tyres cover a broader range across different conditions, used both in moderate temperatures and moderate humidity (~21–26°C, 45–65%).

Wet and Intermediate tyres appear predominantly when air temperature is lower (21–24°C) and humidity is higher (60–85%) — typical of rainy or damp conditions.

The color spread shows that tyre compound choice is heavily influenced by weather conditions:

* Dry compounds dominate in hotter, drier conditions.

* Wet compounds dominate in cooler, wetter conditions.

### Experiement

* How can we predict the optimal lap for a driver to pit during the Monaco Grand Prix based on external features and race conditions?

In order to predict the optimal lap for a driver to pit during the Monaco Grand Prix, we aggregated stint-level race data, engineering several features that capture both driver behavior and external race conditions. By grouping data by Year, Driver, and Stint, and computing statistics like average lap times, lap time variability, stint degradation, track and air temperatures, humidity, and safety car periods, we created a detailed profile of each stint. And we also encoded categorical features such as the initial tire compound and tire era to help the model differentiate between tire types and eras that influence degradation rates.

The target variable, OptimalPitLap, was defined as the lap immediately following the end of a stint, reflecting the logical point where a driver would typically pit under normal race conditions. 

A **Random Forest Regressor** was trained to predict the optimal pit lap using these engineered features. The data was split into 70% train and 30% temp which was was further split into 15% validation and 15% test.

<img width="784" height="384" alt="image" src="https://github.com/user-attachments/assets/610f611a-4e0e-4934-894d-9f80498ed26a" />

	Driver	| Stint	| OptimalPitLap |	PredictedOptimalPitLap
6	RAI	1	18.0	20
51	HAM	3	31.0	30
102	LEC	1	2.0	3
13	VER	1	48.0	48
65	RIC	2	65.0	65
86	PER	3	54.0	44
92	RUS	1	55.0	51
0	ALO	1	20.0	23
44	RIC	2	78.0	63
110	SAI	1	2.0	3
| Driver | Stint | OptimalPitLap | PredictedOptimalPitLap |
| ------ | ----- | ------------- | ---------------------- |
|   RAI  | 1	 |     18.0  	 |       20               |
|   HAM	 | 3	 |     31.0	 |       30               |
|   LEC	 | 1	 |      2.0	 |        3               |
|   VER	 | 1	 |     48.0	 |       48               |
|   RIC	 | 2     |     65.0	 |       65               |
|   PER	 | 3	 |     54.0	 |       44               |
|   RUS	 | 1	 |     55.0	 |       51               |
|   ALO	 | 1	 |     20.0	 |       23               |
|   RIC	 | 2	 |     78.0	 |       63               |
|   SAI	 | 1	 |      2.0	 |        3               |

The model achieved strong performance, with an **R² score** of approximately **0.76** on the validation set and 0.46 on the test set. The **Root Mean Squared Error (RMSE)** was **11.20 laps** on validation data and **12.01 laps** on unseen test data, while the **Mean Absolute Error (MAE)** was 6.77 and 6.16 laps respectively. These results suggest that the model can predict the optimal pit stop lap with reasonable accuracy, especially considering the variability and unpredictability inherent in race conditions.

<img width="857" height="547" alt="image" src="https://github.com/user-attachments/assets/5b5060c0-ea32-4107-9223-b4d5fec728c9" />

<img width="777" height="934" alt="image" src="https://github.com/user-attachments/assets/7e25f8ee-35b6-4e53-b9c2-3183017660b6" />

Identifying the feature importances reveals that the length of the current stint (StintLength) is the most critical factor in determining the optimal pit stop timing. This suggests that the distance covered during a stint strongly influences when a driver should ideally pit. Following that, the maximum stint length observed so far (StintLengthSoFar_max) also plays a significant role, highlighting how the cumulative load on the tires and car over the race affects pit strategy decisions.

Environmental factors continue to be important: the mean air temperature (AirTemp_mean) and the mean track temperature (TrackTemp_mean) are highly ranked. These variables impact tire degradation and overall car performance, indicating that changing atmospheric and track conditions heavily factor into pit decisions. The average lap time (LapTime(s)_mean) also emerged as a major contributor, suggesting that a driver’s consistent pace—or lack thereof—helps signal when a pit stop is necessary.

Interestingly, the driver's change in position on their most recent lap (PositionDelta_last) showed strong influence, hinting that shifts in race competitiveness or overtaking dynamics could trigger strategy changes. Tire-related metrics such as maximum tire life (TyreLife_max) and tire life ratio (TyreLifeRatio) were also critical, reinforcing the importance of managing tire durability over a stint. Humidity (Humidity_mean) and rainfall (Rain_sum) completed the top ten, emphasizing that wet or humid conditions significantly alter the timing of optimal pit stops.

Overall, these results indicate that a blend of stint progression, tire wear management, lap performance, and environmental factors provides a strong and reliable foundation for predicting optimal pit stops during a race.

