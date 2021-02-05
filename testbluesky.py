import bluesky 
bluesky.init("sim-detached")

for i in range(10):
	bluesky.sim.step()
	bluesky.net.step()

print("Ending script...")
