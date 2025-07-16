import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("log.csv")
plt.plot(df["step"], df["centerline_error"])
plt.title("Deviation from centerline over time")
plt.xlabel("Timestep")
plt.ylabel("Error (m)")
plt.grid()
plt.show()
