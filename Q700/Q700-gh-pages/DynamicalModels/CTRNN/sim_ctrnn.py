import ctrnn
import matplotlib.pyplot as plt
import numpy as np

size = 50
duration = 100
stepsize = 0.01

time = np.arange(0.0,duration,stepsize)

nn = ctrnn.CTRNN(size)
nn.randomizeParameters()

nn.initializeState(np.zeros(size))

outputs1 = np.zeros((len(time),size))

# Run simulation
step = 0
for t in time:
    if t > 60:
        nn.Input = np.random.random(size)
    nn.step(stepsize)
    outputs1[step] = nn.Output
    step += 1

# Plot activity
for i in range(size):
    plt.plot(time,outputs1)
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("Neural activity")
plt.show()
