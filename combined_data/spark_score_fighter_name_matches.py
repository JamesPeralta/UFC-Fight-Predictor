from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType


def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
    ''' levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    '''
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indices of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions
    if ratio_calc:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return distance[row][col]


def create_fighter_scores(missing, avail, output):
    spark = SparkSession.builder.master('local').appName('fighter_names') \
        .config('spark.executor.memory', '15g') \
        .config('spark.driver.memory', '15g') \
        .config('spark.executor.instances', '8') \
        .config('spark.executor.cores', '1') \
        .getOrCreate()

    sc = spark.sparkContext

    avail_rdd = sc.parallelize(sc.textFile(avail).map(lambda line: line.strip()).collect())
    missing_rdd = sc.parallelize(sc.textFile(missing).map(lambda line: line.strip()).collect())

    print('Available names: {}'.format(avail_rdd.count()))
    print('Missing names: {}'.format(missing_rdd.count()))

    cartesian = missing_rdd.cartesian(avail_rdd)
    print('Combinations: {}'.format(cartesian.count()))

    cartesian_map = sc.parallelize(cartesian.map(lambda data: {
        'x': data[0],
        'y': data[1],
        'score': int(levenshtein_ratio_and_distance(data[0], data[1])),
        'ratio': float(levenshtein_ratio_and_distance(data[0], data[1], ratio_calc=True)),
    }).collect())

    schema = StructType([
        StructField('x', StringType(), False),
        StructField('y', StringType(), False),
        StructField('score', IntegerType(), False),
        StructField('ratio', FloatType(), False)
    ])

    df = spark.createDataFrame(cartesian_map, schema=schema)
    df.toPandas().to_csv(output, header=True, index=False)
