# Basic parameters
class Params(object):
    DATASET = "uni"

    NDATA = None
    NDIM = None
    LOW = None
    HIGH = None
    nQuery = 2  # number of queries
    unitGrid = 0.01  # cell unit in kd-cell
    ONE_KM = 0.0089982311916  # convert km to degree
    ZIPFIAN_SKEW = 2
    URGENCY_RANDOM = False

    POPULATION_FILE = '../../dataset/gowalla_CA.dat'

    # for grid standard
    # maxHeight = 2
    part_size = 6
    ANALYST_COUNT = 36

    GRID_SIZE = 200

    def __init__(self, seed):
        self.Seed = seed
        self.minPartSize = 2 ** 0  # maximum number of data points in a leaf node

        self.resdir = ""
        self.x_min, self.y_min, self.x_max, self.y_max = None, None, None, None
        self.NDATA = None
        self.NDIM = None
        self.LOW = None
        self.HIGH = None

    def debug(self):
        print self.x_min, self.y_min, self.x_max, self.y_max
        print self.NDATA, self.NDIM, self.LOW, self.HIGH

    def select_dataset(self):
        if Params.DATASET == "mcdonald":
            self.dataset = '../../dataset/mcdonald.dat'
            self.resdir = '../../output/mcdonald/'
            self.x_min = 20.7507
            self.y_min = -159.5862
            self.x_max = 64.8490
            self.y_max = -67.2804
        if Params.DATASET == "uni":
            self.dataset = '../../dataset/la/2015_6_24_SynData_1.dat'
            self.resdir = '../../output/la/'
            self.x_min = 33.976572
            self.y_min = -118.339477
            self.x_max = 34.066572
            self.y_max = -118.229477
        if Params.DATASET == "gau":
            self.dataset = '../../dataset/la/2015_6_24_SynData_2.dat'
            self.resdir = '../../output/la/'
            self.x_min = 33.976572
            self.y_min = -118.339477
            self.x_max = 34.066572
            self.y_max = -118.229477
        if Params.DATASET == "zipf":
            self.dataset = '../../dataset/la/2015_6_24_SynData_3.dat'
            self.resdir = '../../output/la/'
            self.x_min = 33.976572
            self.y_min = -118.339477
            self.x_max = 34.066572
            self.y_max = -118.229477

    # dataset = '../../dataset/gowalla_CA.dat'; resdir = '../../output/gowalla_ca/'
    # x_min=-124.3041; y_min=32.1714; x_max=-114.0043; y_max=41.9984 #gowalla_CA


    # dataset = '../../dataset/tiger_NMWA.dat'; resdir = '../../output/tiger/'
    # x_min=-124.8193; y_min=31.3322; x_max=-103.0020; y_max=49.0025 # tiger

    # dataset = '../../dataset/landmark.dat'; resdir = '../../output/landmark/'
    # x_min=-124.4384; y_min=24.5526; x_max=-67.0255; y_max=49.0016 # landmark

    # dataset = '../../dataset/restrnts.dat'; resdir = '../../output/restrnts/'
    # x_min=-124.4972; y_min=24.5473; x_max=-66.9844; y_max=48.9999 # restrnts

    # dataset = '../../dataset/shopping.dat'; resdir = '../../output/shopping/'
    # x_min=-124.2640; y_min=24.5515; x_max=-68.2106; y_max=48.9939 # shopping

    # dataset = '../../dataset/parkrec.dat'; resdir = '../../output/parkrec/'
    #    x_min=-124.5249; y_min=24.5510; x_max=-66.9687; y_max=49.0010 # parkrec

    #    dataset = '../../dataset/zipcode.dat'; resdir = '../../output/zipcode/'
    #    x_min=-176.6368; y_min=17.9622; x_max=-65.2926; y_max=71.2995 # zipcode

    #    dataset = '../../dataset/truck.dat'; resdir = '../../output/truck/'
    #    x_min=23.5100; y_min=37.8103; x_max=24.0178; y_max=38.2966 # truck

    #    dataset = '../../dataset/ne.dat'; resdir = '../../output/ne/'
    #    x_min=0.0470; y_min=0.0543; x_max=0.9530; y_max=0.9457 # ne

    #    dataset = '../../dataset/na.dat'; resdir = '../../output/na/'
    #    x_min=-174.2057; y_min=14.6805; x_max=-52.7316; y_max=79.9842 # na

    #    dataset = '../../dataset/buses.dat'; resdir = '../../output/buses/'
    #    x_min=22.3331; y_min=37.8329; x_max=24.0203; y_max=38.7417 # buses

    #    dataset = '../../dataset/gowalla_SA.dat'; resdir = '../../output/gowalla_sa/'
    #    x_min=-63.3209; y_min=-176.3086; x_max=13.5330; y_max=-32.4150 # gowalla_SA

    #    dataset = '../../dataset/brightkite.dat'; resdir = '../../output/brightkite/'
    #    x_min=-94.5786; y_min=-179.1541; x_max=90; y_max=178 # brightkite