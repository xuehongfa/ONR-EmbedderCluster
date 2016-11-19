import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def closestpair(L):
	def square(x): return x*x
	def sqdist(p,q): return square(p[0]-q[0])+square(p[1]-q[1])

	# Work around ridiculous Python inability to change variables in outer scopes
	# by storing a list "best", where best[0] = smallest sqdist found so far and
	# best[1] = pair of points giving that value of sqdist.  Then best itself is never
	# changed, but its elements best[0] and best[1] can be.
	#
	# We use the pair L[0],L[1] as our initial guess at a small distance.
	best = [sqdist(L[0],L[1]), (L[0],L[1])]

	# check whether pair (p,q) forms a closer pair than one seen already
	def testpair(p,q):
		d = sqdist(p,q)
		if d < best[0]:
			best[0] = d
			best[1] = p,q

	# merge two sorted lists by y-coordinate
	def merge(A,B):
		i = 0
		j = 0
		while i < len(A) or j < len(B):
			if j >= len(B) or (i < len(A) and A[i][1] <= B[j][1]):
				yield A[i]
				i += 1
			else:
				yield B[j]
				j += 1

	# Find closest pair recursively; returns all points sorted by y coordinate
	def recur(L):
		if len(L) < 2:
			return L
		split = len(L)/2
		splitx = L[split][0]
		L = list(merge(recur(L[:split]), recur(L[split:])))

		# Find possible closest pair across split line
		# Note: this is not quite the same as the algorithm described in class, because
		# we use the global minimum distance found so far (best[0]), instead of
		# the best distance found within the recursive calls made by this call to recur().
		# This change reduces the size of E, speeding up the algorithm a little.
		#
		E = [p for p in L if abs(p[0]-splitx) < best[0]]
		for i in range(len(E)):
			for j in range(1,8):
				if i+j < len(E):
					testpair(E[i],E[i+j])
		return L

	L.sort()
	recur(L)
	return best[1]

vector=open('/home/hongfa/workspace/Embedding/embedding.libsvm','r')
vec = vector.readlines()
X=[]
for v in vec:
    v=v.split(" ")
    new_vector = [0 for i in range(25)]

    for i in range (1,len(v)-1):
        index = int(v[i].split(":")[0])

        feature = float(v[i].split(":")[1])

        for i in range(0,25):
            if index == i:
                new_vector[i]=feature


    #print new_vector

    X.append(new_vector)

T = np.array(X)
#print T
kmeans = KMeans(n_clusters=2)
kmeans.fit(T)
#print T
#Best = closestpair(T)
#print Best
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
#print len(labels)
print(centroids)
#print(labels)
#colors = ["g.","r.","c.","y."]
# pos = 0
# neg = 0


#for i in range(len(T)):
#     #print("coordinate:",T[i], "label:", labels[i])
#     if labels[i] == 0:
#         pos = pos +1
#     else:
#         neg = neg+1
#
#     plt.plot(T[i][0], T[i][1], colors[labels[i]], markersize = 10)
#     ax.annotate(i, xy=(T[i][0],T[i][1]), textcoords='data') # <--
# print pos
# print neg
#
#
#plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
#
#plt.grid()
#plt.show()

reduced_data = PCA(n_components=2).fit_transform(T)
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
fig=plt.figure(1)
ax = fig.add_subplot(111)
for i in range(len(reduced_data)):                                       # <--
    plt.plot(reduced_data[i][0], reduced_data[i][1], 'k.', markersize=4)
    ax.annotate(i, xy=(reduced_data[i][0],reduced_data[i][1]), textcoords='data',size=10)
Best = closestpair(reduced_data)
print reduced_data
print Best
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=20, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the bc benchmark (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()