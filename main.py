import monte_carlo as mc
import temporal_difference as td
import matplotlib.pyplot as plt


# runs 500 episodes and plots results

LENGTH = 500
tdm, tds = td.History(LENGTH)
mcm, mcs = mc.History(LENGTH)


print('mcs: {0}, tds: {1}'.format(mcs, tds))
plt.plot([i[1] for i in mcs])
plt.plot([tdl[1] for tdl in tds])
plt.ylabel('Episode Length')
plt.xlabel('nEpisodes')
plt.legend(('MC, Mean: ' + str(mcm), 'TD, Mean: ' + str(tdm)))
plt.show()
