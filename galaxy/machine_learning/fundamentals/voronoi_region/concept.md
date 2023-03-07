# Concept

The following definition is from [Wikipedia](https://en.wikipedia.org/wiki/Voronoi_diagram).

```{prf:definition} Voronoi Region
:label: def-voronoi-region

Let $X$ be a metric space with distance function $d$. Let $K$ be a set of indices and let $\left(P_k\right)_{k \in K}$ be a tuple (ordered collection) of nonempty subsets (the sites) in the space $X$. The Voronoi cell, or Voronoi region, $R_k$, associated with the site $P_k$ is the set of all points in $X$ whose distance to $P_k$ is not greater than their distance to the other sites $P_j$, where $j$ is any index different from $k$. In other words, if $d(x, A)=\inf \{d(x, a) \mid a \in A\}$ denotes the distance between the point $x$ and the subset $A$, then

$$
R_k=\left\{x \in X \mid d\left(x, P_k\right) \leq d\left(x, P_j\right) \text { for all } j \neq k\right\}
$$

The Voronoi diagram is simply the tuple of cells $\left(R_k\right)_{k \in K}$. In principle, some of the sites can intersect and even coincide (an application is described below for sites representing shops), but usually they are assumed to be disjoint. In addition, infinitely many sites are allowed in the definition (this setting has applications in geometry of numbers and crystallography), but again, in many cases only finitely many sites are considered.

In the particular case where the space is a finite-dimensional Euclidean space, each site is a point, there are finitely many points and all of them are different, then the Voronoi cells are convex polytopes and they can be represented in a combinatorial way using their vertices, sides, two-dimensional faces, etc. Sometimes the induced combinatorial structure is referred to as the Voronoi diagram. In general however, the Voronoi cells may not be convex or even connected.

In the usual Euclidean space, we can rewrite the formal definition in usual terms. Each Voronoi polygon $R_k$ is associated with a generator point $P_k$. Let $X$ be the set of all points in the Euclidean space. Let $P_1$ be a point that generates its Voronoi region $R_1, P_2$ that generates $R_2$, and $P_3$ that generates $R_3$, and so on. Then, all locations in the Voronoi polygon are closer to the generator point of that polygon than any other generator point in the Voronoi diagram in Euclidean plane".
```