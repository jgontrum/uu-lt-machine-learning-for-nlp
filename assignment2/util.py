import collections
import itertools
import sys
import zipfile

import classifier
import matplotlib.pyplot as pyplot
# import nltk
import numpy

if sys.version_info[0] != 3:
    print('Please use Python 3 to run this code.')
    sys.exit(1)


def show_stats(title, log, weights, bias, vocabulary, top_n=10,
               write_to_file="results.csv", configuration=None):
    print(title)
    print()

    best_training_loss = min(l['training_loss_reg'] for l in log)
    best_validation_loss = min(l['val_loss'] for l in log)

    print('Best regularised training loss: %g' % best_training_loss)
    print('Final regularised training loss: %g' % log[-1]['training_loss_reg'])
    print('Best validation loss: %g' % best_validation_loss)
    print('Final validation loss: %g' % log[-1]['val_loss'])

    print()
    print('Number of weights: %d' % len(weights))

    n_large_weights = sum(abs(w) > 0.01 for w in weights)
    print('Bias: %g' % bias)
    print('Number of weights with magnitude > 0.01: %d' % n_large_weights)

    features = list(zip(weights, vocabulary.keys()))
    features.sort()

    print()
    print('Top %d positive features:' % top_n)
    print('\n'.join('%g\t%s' % f for f in sorted(features[-top_n:], reverse=True)))
    print()
    print('Top %d negative features:' % top_n)
    print('\n'.join('%g\t%s' % f for f in features[:top_n]))

    if write_to_file:
        with open(write_to_file, 'a') as f:
            f.write(",".join([
                str(configuration['reg_lambda']),
                str(configuration['learning_rate']),
                str(configuration['loss_function']),
                str(configuration['regulariser']),
                str(configuration['niterations']),
                str(best_training_loss),
                str(log[-1]['training_loss_reg']),
                str(best_validation_loss),
                str(log[-1]['val_loss']),
                str(configuration['val_accuracy'])
            ]) + "\n")


def display_log_record(iteration, log_record):
    print(('Iteration %d: ' % iteration) + ', '.join('%s %g' % (k, v) for k, v in log_record.items()))


def create_plots(title, log, weights, log_keys=None):
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    fig.suptitle(title)
    plot_log(ax1, log, keys=log_keys)
    weight_histogram(ax2, weights)
    pyplot.show()


def plot_log(ax, log, keys=None):
    if keys is None:
        keys = log[0].keys()

    max_loss = 0.0

    for key in keys:
        y = numpy.array([rec[key] for rec in log])
        ax.plot(y, label=key)
        max_loss = max(max_loss, y.max())

    ax.set_ylim(0.0, 1.1 * max_loss)
    ax.set_title('Learning curves')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()


def weight_histogram(ax, weights):
    ax.set_title('Weight histogram')
    ax.set_xlabel('Count')
    ax.set_ylabel('Weight')
    ax.hist(weights, bins=30)


def load_movie_data(infile):
    reviews = []
    labels = []

    with zipfile.ZipFile(infile, 'r') as zf:
        for fname in zf.namelist():
            if not fname.endswith('.txt'):
                continue
            with zf.open(fname, 'r') as f:
                tokens = f.read().decode('utf8').split()
            reviews.append(tokens)
            labels.append(1.0 if fname.startswith('pos') else -1.0)

    return SentimentData('Movie reviews', reviews, labels)


# def load_sentiment_data(infile):
#     sentences = []
#     labels = []
#     with open(infile, 'r', encoding='latin_1') as f:
#         for line in f:
#             fields = line.rstrip('\n').split('\t')
#             if len(fields) != 2:
#                 continue
#             sentence = nltk.tokenize.word_tokenize(fields[0].lower(), language='english')
#             # we follow the book and work with -1/+1
#             label = 1.0 if fields[1] == '1' else -1.0
#             sentences.append(sentence)
#             labels.append(label)
#
#     return SentimentData(os.path.basename(infile), sentences, labels)


class SentimentData:
    def __init__(self, name, sentences, labels):
        self.name = name
        self.sentences = sentences
        self.labels = labels

        unigrams = set()
        bigrams = set()
        for snt in sentences:
            unigrams.update(snt)
            bigrams.update(a + ' ' + b for a, b in zip(snt, snt[1:]))

        self.unigram_vocabulary = collections.OrderedDict((w, i) for i, w in enumerate(sorted(unigrams)))
        self.bigram_vocabulary = collections.OrderedDict((w, i) for i, w in enumerate(sorted(bigrams)))
        self.combined_vocabulary = \
            collections.OrderedDict((w, i) for i, w in enumerate(itertools.chain(sorted(unigrams), sorted(bigrams))))

        self.feature_type = 'unigram'
        self.vocabulary = self.unigram_vocabulary

    def __len__(self):
        return len(self.sentences)

    def select_feature_type(self, features):
        available_types = {
            'unigram': self.unigram_vocabulary,
            'bigram': self.bigram_vocabulary,
            'unigram+bigram': self.combined_vocabulary
        }
        self.feature_type = features
        self.vocabulary = available_types[features]

    def random_split(self, proportions):
        nexamples = len(self.sentences)
        sum_p = sum(proportions)
        idx = numpy.cumsum(numpy.array([0] + [int(p * nexamples / sum_p) for p in proportions], dtype=numpy.int32))
        perm = numpy.random.permutation(nexamples)
        return self._split_data(idx, perm)

    def train_val_test_split(self):
        idx = numpy.array([0, 1600, 1800, 2000], dtype=numpy.int32)
        perm = numpy.array([1051, 1537, 864, 855, 1080, 120, 1652, 446, 1180, 1070, 375,
                            936, 1645, 1188, 759, 1772, 1864, 1716, 654, 1220, 646, 1669,
                            1616, 1728, 361, 1279, 1717, 350, 492, 853, 134, 1089, 1572,
                            353, 1835, 1009, 444, 551, 923, 625, 24, 792, 1095, 441,
                            46, 1744, 871, 1824, 1346, 1169, 826, 945, 1942, 974, 662,
                            1393, 1251, 1084, 1312, 1926, 1982, 926, 1566, 452, 1943, 1995,
                            1589, 562, 1940, 938, 146, 801, 1900, 803, 757, 433, 1481,
                            1470, 1343, 460, 1891, 942, 439, 1598, 996, 1794, 1625, 1657,
                            305, 827, 1241, 1133, 741, 1121, 799, 453, 1467, 183, 595,
                            1107, 1424, 1024, 398, 834, 145, 1838, 820, 1979, 1832, 147,
                            1613, 1950, 1977, 1064, 1283, 399, 1083, 747, 1282, 1738, 1797,
                            194, 1640, 1847, 1418, 140, 1131, 316, 1870, 1452, 1331, 1564,
                            560, 1016, 795, 346, 1227, 1104, 1490, 1019, 246, 77, 1650,
                            1829, 1802, 932, 328, 752, 469, 1047, 1857, 809, 243, 754,
                            1745, 207, 263, 1727, 1501, 303, 1595, 1286, 1140, 166, 1608,
                            1868, 1994, 1048, 918, 1139, 756, 1712, 96, 292, 508, 481,
                            1707, 1676, 1851, 369, 385, 1636, 1859, 1028, 1493, 52, 178,
                            1705, 1773, 916, 1356, 1539, 234, 1865, 1837, 1454, 745, 644,
                            17, 1332, 509, 1917, 796, 1231, 1850, 149, 340, 821, 200,
                            532, 1952, 1503, 1294, 1097, 366, 256, 1421, 736, 651, 1149,
                            317, 1336, 1610, 267, 415, 1964, 1137, 989, 219, 708, 1032,
                            104, 191, 1003, 1916, 1449, 293, 307, 1953, 706, 1630, 925,
                            713, 1320, 987, 1664, 507, 266, 1017, 1526, 1882, 404, 1295,
                            411, 1871, 1392, 1924, 951, 1741, 816, 1991, 639, 656, 1448,
                            543, 1734, 1671, 1775, 155, 1430, 545, 667, 1538, 28, 512,
                            522, 1091, 1725, 1853, 1081, 1201, 176, 1374, 810, 576, 212,
                            1629, 434, 343, 531, 335, 49, 1031, 688, 1848, 601, 215,
                            1280, 882, 13, 823, 591, 1622, 1723, 739, 387, 955, 704,
                            1908, 689, 1276, 1686, 1778, 282, 414, 655, 813, 793, 1791,
                            40, 1623, 468, 1591, 1146, 1177, 1654, 807, 618, 1310, 1063,
                            782, 1812, 1160, 1007, 252, 11, 1152, 1135, 1906, 1004, 724,
                            478, 1528, 336, 363, 1494, 765, 156, 238, 448, 1396, 1574,
                            1094, 378, 1305, 1321, 774, 1219, 1584, 2, 405, 406, 897,
                            1679, 1769, 1634, 475, 911, 1194, 1531, 616, 1404, 410, 184,
                            579, 493, 1788, 804, 839, 280, 144, 1578, 953, 1514, 1787,
                            1216, 152, 233, 1607, 1373, 1428, 1617, 842, 1261, 162, 526,
                            1603, 271, 513, 14, 1469, 1967, 1525, 59, 245, 368, 32,
                            1427, 776, 278, 257, 1214, 521, 1580, 1961, 652, 1192, 1408,
                            459, 1936, 1109, 69, 852, 1674, 506, 841, 1704, 1461, 1569,
                            1191, 964, 544, 349, 992, 1588, 1651, 133, 1724, 171, 1168,
                            1371, 1098, 1527, 1890, 291, 117, 1754, 697, 718, 969, 284,
                            726, 1912, 1000, 1861, 819, 883, 1391, 1412, 1811, 1112, 295,
                            1499, 348, 186, 1437, 109, 1683, 1762, 121, 948, 848, 1903,
                            172, 1799, 1951, 769, 873, 1815, 1126, 1074, 1497, 327, 1750,
                            1368, 901, 1680, 48, 749, 702, 1731, 536, 913, 1521, 1687,
                            373, 1970, 1050, 1846, 1205, 1472, 863, 1347, 755, 273, 1399,
                            482, 877, 1101, 1175, 374, 30, 636, 424, 1386, 276, 649,
                            525, 102, 1275, 362, 1127, 1262, 294, 1694, 972, 1202, 666,
                            552, 1417, 318, 1463, 106, 1808, 1947, 742, 1760, 553, 159,
                            1684, 1193, 124, 1597, 1010, 683, 1100, 1915, 895, 359, 1023,
                            511, 1756, 1376, 1699, 1489, 91, 313, 1587, 1855, 1902, 1546,
                            1810, 99, 924, 570, 712, 1976, 1960, 1059, 1345, 75, 1102,
                            599, 1120, 573, 590, 168, 457, 1092, 1255, 1786, 22, 1480,
                            1582, 686, 1568, 1945, 1896, 607, 1698, 1800, 558, 1758, 1066,
                            1785, 1349, 954, 100, 1330, 1875, 261, 657, 8, 1618, 1957,
                            1522, 1596, 428, 67, 1008, 1512, 1380, 1184, 1243, 717, 1301,
                            1001, 665, 1946, 1353, 1611, 358, 1316, 1409, 1624, 624, 29,
                            1507, 175, 391, 1040, 394, 1246, 1327, 556, 1941, 179, 1806,
                            222, 1155, 1831, 614, 1210, 36, 995, 55, 1256, 1420, 1106,
                            1844, 1445, 1033, 1978, 1771, 677, 703, 1726, 1763, 1366, 835,
                            421, 523, 696, 859, 1436, 1872, 857, 1429, 1556, 1974, 1761,
                            458, 760, 891, 1781, 98, 748, 1555, 598, 1400, 1473, 94,
                            979, 167, 1856, 628, 204, 984, 310, 500, 1164, 1457, 485,
                            455, 1344, 921, 226, 844, 1372, 1468, 1432, 1822, 1969, 746,
                            388, 893, 1751, 449, 74, 840, 634, 157, 1939, 1476, 1817,
                            1973, 784, 403, 139, 546, 788, 671, 1041, 324, 1093, 705,
                            1635, 135, 1558, 1118, 988, 1554, 1681, 1034, 1805, 719, 1739,
                            1714, 400, 196, 262, 496, 1260, 962, 678, 205, 1833, 633,
                            93, 23, 593, 1111, 461, 39, 1968, 1874, 103, 88, 1553,
                            929, 1425, 641, 581, 637, 910, 356, 554, 817, 884, 947,
                            25, 322, 126, 908, 1460, 309, 1570, 154, 1355, 1920, 1271,
                            1300, 198, 501, 1030, 1411, 57, 638, 1515, 440, 1132, 1779,
                            830, 1158, 118, 417, 382, 379, 1440, 143, 1905, 1303, 467,
                            1793, 541, 224, 1044, 413, 1743, 1414, 1956, 1742, 82, 1416,
                            1675, 1183, 1600, 473, 577, 1378, 642, 1543, 977, 1839, 900,
                            1415, 1782, 1035, 1532, 1852, 720, 1904, 1108, 471, 131, 606,
                            484, 423, 1475, 1631, 337, 1354, 675, 1627, 211, 1388, 679,
                            1042, 886, 750, 1881, 1914, 845, 959, 831, 1163, 727, 524,
                            1626, 79, 477, 1453, 1198, 7, 1110, 1253, 1402, 1573, 272,
                            113, 35, 412, 1826, 296, 1790, 1204, 640, 320, 231, 1078,
                            1700, 244, 185, 1426, 1150, 899, 1759, 50, 1732, 1682, 658,
                            164, 1701, 919, 1039, 376, 372, 1605, 1567, 180, 1236, 1141,
                            177, 1350, 1599, 1721, 1077, 592, 1894, 1998, 564, 1082, 691,
                            289, 1447, 498, 1403, 288, 1748, 1206, 297, 465, 483, 1335,
                            534, 290, 1465, 1586, 1885, 832, 270, 1755, 21, 1757, 1406,
                            112, 1299, 767, 661, 229, 1065, 42, 1581, 150, 435, 1339,
                            1511, 1484, 1561, 26, 393, 700, 580, 1381, 559, 958, 635,
                            1820, 38, 1695, 1054, 170, 1383, 173, 621, 338, 1013, 451,
                            218, 1646, 1653, 342, 1237, 1633, 1492, 1385, 1314, 1328, 1517,
                            85, 1990, 1713, 1157, 45, 18, 1668, 622, 1620, 130, 1648,
                            431, 1496, 710, 1015, 805, 384, 123, 941, 1585, 1242, 1087,
                            1665, 514, 997, 965, 1226, 1954, 1211, 462, 1189, 377, 73,
                            491, 1229, 1377, 998, 1736, 1011, 517, 1888, 1843, 885, 1730,
                            963, 1639, 565, 1649, 619, 1056, 1576, 838, 904, 1230, 72,
                            1642, 557, 1696, 1329, 1076, 930, 129, 1340, 1545, 881, 1935,
                            1500, 1571, 837, 128, 221, 791, 274, 1541, 1563, 466, 1151,
                            1689, 960, 1269, 582, 114, 1257, 604, 764, 530, 1006, 786,
                            31, 1395, 915, 550, 1043, 1506, 1359, 502, 1174, 850, 1136,
                            946, 1677, 488, 105, 161, 1022, 1884, 1272, 1886, 1495, 1693,
                            1663, 1021, 878, 101, 931, 985, 1333, 1287, 301, 1659, 561,
                            645, 6, 1938, 744, 725, 772, 1185, 236, 418, 425, 888,
                            515, 990, 182, 674, 978, 1434, 1628, 1061, 1166, 1559, 1747,
                            1703, 1710, 163, 542, 364, 1937, 1387, 1365, 1218, 1145, 1221,
                            1277, 1560, 1296, 51, 732, 407, 201, 766, 1718, 1819, 136,
                            833, 1382, 1549, 142, 151, 1929, 1384, 1394, 1410, 37, 476,
                            1999, 1311, 86, 1002, 1590, 1114, 1533, 1796, 1479, 870, 903,
                            339, 1604, 1143, 192, 1068, 967, 78, 780, 1178, 981, 1753,
                            889, 1892, 381, 110, 733, 325, 1845, 34, 383, 836, 235,
                            673, 956, 1784, 1309, 1764, 1647, 321, 1691, 617, 1536, 1766,
                            1361, 1678, 195, 957, 61, 935, 1638, 1170, 202, 600, 241,
                            1477, 1823, 585, 743, 1324, 1931, 1072, 1854, 1153, 1154, 1818,
                            1302, 892, 1661, 352, 1523, 1351, 312, 1090, 214, 84, 824,
                            1821, 1268, 1443, 1401, 1413, 1513, 812, 1292, 1981, 213, 740,
                            237, 912, 1792, 1895, 1709, 1909, 3, 734, 115, 450, 535,
                            1552, 672, 1232, 1948, 587, 153, 286, 548, 1215, 390, 1765,
                            1656, 1930, 818, 693, 858, 193, 251, 314, 1509, 1099, 445,
                            1863, 300, 1435, 1594, 519, 698, 539, 1124, 1086, 242, 1057,
                            1062, 1455, 1752, 1159, 983, 1225, 370, 1504, 436, 1615, 516,
                            711, 890, 1577, 527, 299, 341, 875, 699, 209, 1207, 798,
                            1398, 1919, 1238, 966, 1105, 781, 239, 1379, 1621, 1867, 1768,
                            43, 973, 1285, 181, 430, 753, 1612, 1667, 132, 1313, 281,
                            1058, 676, 1841, 1795, 758, 629, 1986, 1200, 1212, 1182, 255,
                            160, 1451, 401, 971, 420, 76, 856, 247, 311, 802, 1955,
                            800, 874, 1804, 354, 1491, 1352, 1055, 54, 1827, 794, 1176,
                            692, 1575, 785, 1441, 1466, 60, 1893, 1534, 778, 1869, 1746,
                            116, 695, 865, 1803, 1165, 612, 1326, 1370, 250, 1505, 968,
                            1203, 127, 777, 1458, 389, 961, 1045, 1362, 1641, 429, 1049,
                            991, 1842, 602, 1341, 326, 1535, 1720, 456, 1898, 952, 15,
                            1958, 605, 887, 690, 1197, 869, 65, 1557, 1267, 668, 12,
                            1606, 761, 1179, 1592, 768, 1901, 623, 723, 227, 1258, 1397,
                            1985, 751, 223, 1996, 427, 1918, 1643, 333, 1128, 722, 298,
                            610, 264, 1887, 19, 730, 258, 1866, 1060, 107, 1520, 659,
                            505, 1222, 4, 770, 1602, 586, 1036, 487, 470, 594, 494,
                            1423, 490, 1172, 1899, 1186, 1208, 1542, 1897, 53, 735, 1883,
                            1125, 1749, 1291, 454, 357, 851, 1052, 1836, 1776, 165, 626,
                            90, 1167, 1317, 395, 371, 64, 1085, 1419, 1959, 715, 811,
                            1266, 225, 438, 664, 1263, 902, 1358, 1334, 158, 419, 1067,
                            1181, 260, 714, 1830, 1922, 1442, 216, 682, 814, 249, 20,
                            1666, 917, 1483, 681, 905, 87, 1934, 1053, 670, 1486, 1363,
                            1315, 1583, 497, 896, 1670, 653, 1688, 169, 1113, 1319, 302,
                            1249, 1433, 631, 1880, 1037, 797, 1662, 1814, 228, 232, 397,
                            208, 914, 643, 762, 1878, 1975, 1239, 1014, 83, 1798, 1993,
                            1235, 1737, 660, 866, 880, 1502, 828, 529, 323, 685, 189,
                            976, 1932, 1963, 1828, 1252, 437, 426, 1690, 982, 994, 1223,
                            1471, 1122, 206, 70, 1529, 589, 829, 1825, 1195, 203, 1307,
                            1147, 1673, 1927, 680, 187, 1129, 603, 1530, 447, 1134, 1364,
                            1508, 1949, 1498, 999, 1879, 442, 1706, 806, 141, 1928, 993,
                            1873, 1217, 1987, 1196, 1116, 5, 1284, 1722, 1637, 1281, 1729,
                            920, 1254, 663, 1130, 648, 1544, 1840, 563, 927, 1156, 1783,
                            503, 138, 933, 16, 1431, 566, 537, 355, 1027, 1780, 1849,
                            92, 975, 1450, 620, 518, 1290, 771, 731, 1911, 1540, 344,
                            277, 694, 1375, 578, 632, 268, 1547, 81, 1444, 1360, 1142,
                            1071, 122, 609, 217, 44, 1162, 58, 360, 0, 486, 1462,
                            538, 1079, 1123, 1025, 1910, 279, 790, 1933, 10, 1711, 41,
                            329, 1980, 1672, 1308, 549, 572, 480, 1913, 66, 1813, 254,
                            422, 1740, 1213, 533, 1809, 259, 240, 970, 504, 1171, 775,
                            1020, 47, 1984, 687, 1088, 1801, 1655, 1609, 1518, 269, 443,
                            1834, 1482, 1562, 95, 728, 879, 1234, 1965, 1658, 773, 934,
                            861, 1889, 849, 909, 1550, 1770, 1702, 555, 1860, 365, 351,
                            1487, 148, 474, 867, 1274, 432, 188, 1389, 939, 894, 1972,
                            1579, 567, 986, 1367, 1551, 479, 1247, 89, 569, 1971, 1288,
                            1944, 1224, 684, 56, 1877, 1733, 1245, 489, 1921, 738, 137,
                            1298, 402, 210, 408, 1248, 1264, 174, 1306, 583, 308, 306,
                            1715, 1593, 347, 332, 319, 574, 815, 1474, 63, 1735, 386,
                            1304, 729, 1405, 220, 1923, 68, 125, 846, 1026, 1390, 647,
                            1293, 1989, 1322, 1422, 1342, 1338, 1459, 1348, 1161, 416, 1644,
                            97, 1273, 822, 1464, 1992, 1297, 571, 1907, 669, 345, 1233,
                            949, 1708, 1265, 1117, 1446, 1719, 1478, 1369, 367, 568, 1488,
                            380, 862, 1075, 1524, 1018, 1632, 334, 392, 331, 584, 613,
                            763, 907, 199, 630, 1289, 1038, 285, 1685, 1439, 611, 197,
                            928, 779, 248, 1244, 1138, 980, 1103, 510, 1250, 1692, 464,
                            860, 1173, 1199, 944, 596, 650, 472, 315, 1601, 721, 1,
                            1278, 265, 547, 716, 847, 330, 937, 1318, 1767, 1323, 1096,
                            701, 1069, 1046, 707, 1485, 808, 1966, 1005, 1456, 27, 854,
                            1259, 940, 737, 1438, 709, 499, 1548, 1115, 230, 787, 1660,
                            1510, 1997, 1988, 1816, 1697, 62, 108, 33, 1337, 1012, 876,
                            1073, 1187, 1029, 1565, 1148, 872, 304, 520, 868, 1270, 843,
                            1190, 575, 588, 9, 253, 1619, 275, 1789, 789, 906, 396,
                            1876, 950, 1228, 1357, 540, 1614, 1983, 1858, 1862, 409, 1119,
                            463, 783, 80, 1407, 1325, 1144, 71, 1519, 825, 1209, 1807,
                            608, 190, 111, 528, 922, 287, 1240, 1962, 495, 119, 1774,
                            1516, 627, 1777, 283, 597, 1925, 898, 943, 615])

        return self._split_data(perm, idx)

    def _split_data(self, perm, idx):
        split = []
        for i in range(len(idx) - 1):
            sub_sentences = [self.sentences[j] for j in perm[idx[i]:idx[i + 1]]]
            sub_labels = [self.labels[j] for j in perm[idx[i]:idx[i + 1]]]
            subset = SentimentData('%s_%d' % (self.name, i), sub_sentences, sub_labels)
            subset.unigram_vocabulary = self.unigram_vocabulary
            subset.bigram_vocabulary = self.bigram_vocabulary
            subset.combined_vocabulary = self.combined_vocabulary
            subset.select_feature_type(self.feature_type)
            split.append(subset)
        return split

    def features(self):
        for snt in self.sentences:
            if self.feature_type == 'unigram':
                unigrams = {self.unigram_vocabulary[w] for w in snt}
                yield unigrams
            elif self.feature_type == 'bigram':
                bigrams = {self.bigram_vocabulary[a + ' ' + b] for a, b in zip(snt, snt[1:])}
                yield bigrams
            elif self.feature_type == 'unigram+bigram':
                unigrams = {self.unigram_vocabulary[w] for w in snt}
                bigrams = {self.bigram_vocabulary[a + ' ' + b] for a, b in zip(snt, snt[1:])}
                yield unigrams | bigrams
            else:
                raise ValueError('Unknown feature type: ' + self.feature_type)


class TestSentimentData:
    def __init__(self):
        self.labels = [1.0, -1.0, 1.0, 1.0, -1.0]
        self.data = [
            {0, 5, 12, 13, 18, 34, 36, 48},
            {2, 3, 8, 19},
            {7, 29, 35, 36, 37, 49},
            {0, 1, 2, 3, 4, 5, 6, 7},
            {41, 42, 43, 44, 45, 46, 47, 48, 49}
        ]

    def __len__(self):
        return len(self.labels)

    def features(self):
        return self.data


def run_tests():
    test_gradient_descent_step()
    test_hinge_loss()
    test_l2_regularisation()


def test_gradient_descent_step():
    old_weights = [-0.64785033, -0.77454901, -1.33731302, 0.92178561, 0.44660898,
                   -0.63342684, 0.37721042, 0.60969772, 1.03884045, -0.10405732,
                   0.27633419, -1.77313231, -1.76624382, 0.88037165, -1.4688971,
                   0.40207157, -1.32744739, -0.37214431, -1.91257454, 0.46648012,
                   2.01194033, 0.48938136, -0.20018781, -0.05424772, 0.56570139,
                   -0.1643061, 0.70487633, 2.17959453, 0.55962729, 1.89147588,
                   0.27310441, -1.95814671, 0.55295444, 1.4118642, 0.07533605,
                   -0.45240706, 0.59262564, 0.60791175, -0.37885029, 0.93890843,
                   1.18156882, 0.00654753, 0.83082654, -0.38465219, -0.49886363,
                   1.6415025, 1.10697724, -1.50496406, 0.49031257, 1.65183362]
    old_bias = -0.312345

    weight_grads = [0.05734476, 0.03513378, 0.03488778, -0.10978203, 0.0654562,
                    0.09329271, 0.02304053, 0.07992804, -0.0899072, 0.07387878,
                    -0.08164659, 0.20069353, 0.14469181, -0.0600909, 0.15584801,
                    0.02537515, 0.24089286, -0.04409187, -0.08106724, 0.06451122,
                    0.08685967, 0.00750409, -0.03671001, 0.02968672, 0.22221776,
                    -0.02084546, 0.17695168, -0.03489778, -0.05734652, -0.04087372,
                    0.03032844, -0.04312753, -0.07807858, -0.11582963, -0.00227852,
                    0.040502, -0.16227115, 0.18737644, -0.17028486, -0.00601666,
                    -0.03892575, -0.09198075, 0.14414415, -0.10946207, -0.00855337,
                    0.01641435, 0.16609624, -0.076496, 0.05982559, 0.14761389]
    bias_grad = 0.023413

    new_weights = [-0.65550157, -0.77923674, -1.34196793, 0.93643331, 0.43787547,
                   -0.64587445, 0.37413623, 0.59903329, 1.05083634, -0.11391462,
                   0.28722791, -1.7999099, -1.78554937, 0.8883893, -1.48969117,
                   0.39868588, -1.35958859, -0.36626134, -1.90175811, 0.45787269,
                   2.00035105, 0.48838013, -0.19528977, -0.05820868, 0.53605192,
                   -0.16152479, 0.6812665, 2.18425077, 0.56727876, 1.89692947,
                   0.26905783, -1.9523924, 0.56337209, 1.42731881, 0.07564007,
                   -0.45781105, 0.61427672, 0.58291099, -0.35612999, 0.93971121,
                   1.1867625, 0.01882009, 0.81159406, -0.37004718, -0.4977224,
                   1.63931241, 1.08481579, -1.49475756, 0.48233032, 1.6321382]
    new_bias = -0.3154688865489

    lr = 0.1334253

    upd_weights, upd_bias = classifier.gradient_descent_step(lr, old_weights, old_bias,
                                                     weight_grads, bias_grad)

    if not numpy.allclose(upd_weights, new_weights):
        print('FAILED gradient descent test 1: Got incorrect weight vector.')
    else:
        print('PASSED gradient descent test 1: Got correct weight vector.')

    if not numpy.isclose(upd_bias, new_bias):
        print('FAILED gradient descent test 2: Got incorrect bias term.')
    else:
        print('PASSED gradient descent test 2: Got correct bias term.')


def test_hinge_loss():
    weights = [-0.65550157, -0.77923674, -1.34196793, 0.93643331, 0.43787547,
               -0.64587445, 0.37413623, 0.59903329, 1.05083634, -0.11391462,
               0.28722791, -1.7999099, -1.78554937, 0.8883893, -1.48969117,
               0.39868588, -1.35958859, -0.36626134, -1.90175811, 0.45787269,
               2.00035105, 0.48838013, -0.19528977, -0.05820868, 0.53605192,
               -0.16152479, 0.6812665, 2.18425077, 0.56727876, 1.89692947,
               0.26905783, -1.9523924, 0.56337209, 1.42731881, 0.07564007,
               -0.45781105, 0.61427672, 0.58291099, -0.35612999, 0.93971121,
               1.1867625, 0.01882009, 0.81159406, -0.37004718, -0.4977224,
               1.63931241, 1.08481579, -1.49475756, 0.48233032, 1.6321382]
    bias = 3.3154688865489

    loss_fn = classifier.HingeLoss()
    data = TestSentimentData()

    loss = loss_fn.unregularised_loss(weights, bias, data)
    weight_grads, bias_grad = loss_fn.gradients(weights, bias, data)

    target_loss = 2.73063482330978
    target_weight_grads = [-0.2, 0.0, 0.2, 0.2, 0.0,
                           -0.2, 0.0, 0.0, 0.2, 0.0,
                           0.0, 0.0, -0.2, -0.2, 0.0,
                           0.0, 0.0, 0.0, -0.2, 0.2,
                           0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, -0.2,
                           0.0, -0.2, 0.0, 0.0, 0.0,
                           0.0, 0.2, 0.2, 0.2, 0.2,
                           0.2, 0.2, 0.2, 0.0, 0.2]
    target_bias_grad = 0.2

    if not numpy.isclose(loss, target_loss):
        print('FAILED hinge loss test 1: Got incorrect loss value.')
    else:
        print('PASSED hinge loss test 1: Got correct loss value.')

    if not numpy.allclose(weight_grads, target_weight_grads):
        print('FAILED hinge loss test 2: Got incorrect weight gradient.')
    else:
        print('PASSED hinge loss test 2: Got correct weight gradient.')

    if not numpy.isclose(bias_grad, target_bias_grad):
        print('FAILED hinge loss test 2: Got incorrect bias gradient.')
    else:
        print('PASSED hinge loss test 2: Got correct bias gradient.')


def test_l2_regularisation():
    weights = numpy.array([-0.65550157, -0.77923674, -1.34196793, 0.93643331, 0.43787547,
                           -0.64587445, 0.37413623, 0.59903329, 1.05083634, -0.11391462,
                           0.28722791, -1.7999099, -1.78554937, 0.8883893, -1.48969117,
                           0.39868588, -1.35958859, -0.36626134, -1.90175811, 0.45787269,
                           2.00035105, 0.48838013, -0.19528977, -0.05820868, 0.53605192,
                           -0.16152479, 0.6812665, 2.18425077, 0.56727876, 1.89692947,
                           0.26905783, -1.9523924, 0.56337209, 1.42731881, 0.07564007,
                           -0.45781105, 0.61427672, 0.58291099, -0.35612999, 0.93971121,
                           1.1867625, 0.01882009, 0.81159406, -0.37004718, -0.4977224,
                           1.63931241, 1.08481579, -1.49475756, 0.48233032, 1.6321382])

    regulariser = classifier.L2Regulariser()
    loss = regulariser.loss(weights)
    grads = regulariser.gradients(weights)

    target_loss = 27.402342853229715

    if numpy.isclose(loss, target_loss):
        print('PASSED l2 regularisation test 1: Got correct loss value: R(w,b) = 0.5 * ||w||^2.')
        if not numpy.allclose(grads, weights):
            print('FAILED l2 regularisation test 2: Got incorrect weight gradient.')
        else:
            print('PASSED l2 regularisation test 2: Got correct weight gradient.')
    elif numpy.isclose(loss, 2.0 * target_loss):
        print('PASSED l2 regularisation test 1: Got correct loss value: R(w,b) = ||w||^2.')
        if not numpy.allclose(grads, 2.0 * weights):
            print('FAILED l2 regularisation test 2: Got incorrect weight gradient.')
        else:
            print('PASSED l2 regularisation test 2: Got correct weight gradient.')
    else:
        print('FAILED l2 regularisation test 1: Got incorrect loss value.')
        print('SKIPPED l2 regularisation test 2: Please fix loss value first.')

