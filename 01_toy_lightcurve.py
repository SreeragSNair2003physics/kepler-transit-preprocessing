import matplotlib.pyplot as plt

days= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]                                                       # lists when each measurement was made
brightness= [100, 100, 99, 100, 100, 100, 99, 100, 100, 100, 99, 100]                               # lists what Kepler-like measurements said at those times

dip_days= []

for day, value in zip(days, brightness):            # Checks the brightness on each day; on the paired entries in zip

    if value<100:
        dip_days.append(day)                        # Adds the day where the if condition is true

print("\nDip days:", dip_days)
print("First gap:", dip_days[1] - dip_days[0])             
print("Second gap:", dip_days[2] - dip_days[1])   

period = dip_days[1] - dip_days[0]                  # The spacing b/n dips is called period
next_dip = dip_days[-1] + period                    # Predicting next dip after last dip

print("\nPeriod:", period, "days")
print("Predicted next dip:", next_dip)

usual_brightness = 100                              # Records the normal level we chose for this made-up star
lowest_brightness = min(brightness)                 # 	Finds the smallest number in the brightness list
depth = usual_brightness - lowest_brightness

print("\nDip depth:", depth)

first_dip_day = dip_days[0]
print("\nReference dip day:", first_dip_day)




plt.plot(days, brightness, marker="o")
plt.xlabel("Day")
plt.ylabel("Brightness")
plt.title("A made-up star's light curve")
plt.savefig("toy_lightcurve.png")
plt.show()