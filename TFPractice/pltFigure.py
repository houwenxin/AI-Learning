import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 50)
y1 = 3 * x + 2
y2 = x ** 2

# Create the first figure
plt.figure(num=1, figsize=(7, 6))
plt.plot(x, y1)
plt.plot(x, y2, color="red", linewidth=3.0, linestyle="--")

# Create the second figure
plt.figure(num=2)
plt.plot(x, y2, color="green")
# Show
plt.show()
