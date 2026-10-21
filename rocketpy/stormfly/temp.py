import pandas as pd

df = pd.read_csv("rocketpy/stormfly/drag.csv")
print(df)
df = df[['MachNumber', 'DragCoefficient']]
df.to_csv("rocketpy/stormfly/swapped_drag.csv", index=False)