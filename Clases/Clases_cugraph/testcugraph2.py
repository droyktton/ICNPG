import cugraph
import cudf

# read data into a cuDF DataFrame using read_csv
#gdf = cudf.read_csv("graph_data.csv", names=["src", "dst"], dtype=["int32", "int32"])
gdf = cudf.read_csv("https://raw.githubusercontent.com/droyktton/data/main/graph_data.csv")

# pesos (necesarios para algunas cosas nomas)
gdf["data"] = 1.0


# We now have data as edge pairs
# create a Graph using the source (src) and destination (dst) vertex pairs
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='src', destination='dst')

print("\n\n")



######################################
#https://es.wikipedia.org/wiki/PageRank
#
# PR(A) = (1-d)+d Sum_{i=1}^n PR(i)/C(i)
#
# A es el PageRank de la página A.
# d es un factor de amortiguación que tiene un valor entre 0 y 1.
# PR(i)son los valores de PageRank que tienen cada una de las páginas i que enlazan a A.
# C(i) es el número total de enlaces salientes de la página i (sean o no hacia A).

# Let's now get the PageRank score of each vertex by calling cugraph.pagerank
df_page = cugraph.pagerank(G)

# Let's look at the PageRank Score (only do this on small graphs)
for i in range(len(df_page)):
	print("vertex " + str(df_page['vertex'].iloc[i]) +
		" PageRank is " + str(df_page['pagerank'].iloc[i]))

print("\n\n")

#####################################
#https://en.wikipedia.org/wiki/Component_(graph_theory)

# Call cugraph.weakly_connected_components on the dataframe
df_cc = cugraph.weakly_connected_components(G)

# Use groupby on the 'labels' column of the WCC output to get the counts of each connected component label
label_gby = df_cc.groupby('labels')
label_count = label_gby.count()

print("Total number of components found : ", len(label_count))

# Call nlargest on the groupby result to get the row where the component count is the largest
largest_component = label_count.nlargest(n = 1, columns = 'vertex')
print("Size of the largest component is found to be : ", largest_component['vertex'].iloc[0])

smallest_component = label_count.nsmallest(n = 1, columns = 'vertex')
print("Size of the smallest component is found to be : ", smallest_component['vertex'].iloc[0])

#df_cc.head(20)

print("\n\n")

#############################################################
# https://en.wikipedia.org/wiki/Shortest_path_problem
# Call cugraph.sssp to get the distances from vertex 1:

df_ssp = cugraph.sssp(G, 1)

print("no todos los caminos conducen al nodo 1, y estos son los minimos:")

# Print the paths
for index, row in df_ssp.to_pandas().iterrows():
    v = int(row['vertex'])
    p = cugraph.utils.get_traversed_path_list(df_ssp, v)
    print(v, ': ', p)

print("\n\n")

