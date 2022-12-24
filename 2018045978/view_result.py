from matplotlib import pyplot as plt


with open("result_msg.txt", "r") as f:
    lines = f.readlines()
    
    
plotx, ploty = [], []
for line in lines:
    splited_line = line.split()
    episode = int(splited_line[1])
    live_time = int(float(splited_line[4]))
    plotx.append(episode)
    ploty.append(live_time)
    

plt.plot(plotx, ploty, "r-")
plt.xlabel("Episode")
plt.ylabel("Time Steps")
plt.show()