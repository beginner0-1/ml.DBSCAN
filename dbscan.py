import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

x,y = make_moons(
    n_samples=500,
    noise= 0.1,
    random_state =42
)

sclaer = StandardScaler()
x_scaled = sclaer.fit_transform(x)

sns.scatterplot(x=x_scaled[:,0],y = x_scaled[:,1])

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)
labels = dbscan.fit_predict(x_scaled)