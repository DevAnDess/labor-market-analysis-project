from clusterization.k_means.kaagle_salary import kaagle_salary
from clusterization.k_means.hh_salary import hh_salary
from clusterization.hierarchical.kaagle_salary_hierarchical import kaagle_salary_hierarchical

def clusters_main():
    kaagle_salary()
    hh_salary()
    kaagle_salary_hierarchical()

if __name__ == "__main__":
    clusters_main()