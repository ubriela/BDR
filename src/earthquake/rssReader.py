""" rssReader.py reads given earthquake rss (Atom). If there is a new earthquake ID, the new event is added
    to the dictionary and its corresponding information is populated.
    Based on a threshold for mmi, the MBR corresponding to that threshold is assigned to each event
    This MBR can be used as a task region."""

""" Known bugs: if the earthquake is expanded in lon 180 to lon -180 area, the provided MBR is wrong.
    This case needs to be handled later."""
__author__ = "Sasan Tavakkol"
__email__ = "tavakkol@usc.edu"
__date__ = "10/30/2015"


import feedparser
import urllib2
import cStringIO
import zipfile
import threading

#EarthquakeEvents is a dictionary to store events.
#Keys are earthquake ID's and values are EarthqaukeEvent's
earthqaukeEvents = {} #This can be loaded from a file when the code starts


THRESHOLD_MMI = 5 #This is threshold for earthquake intensity to create the task region. 
POLLING_INTERVAL = 5*60 # Interval for polling rss feed in seconds.
silent = False
######## HELPER CLASSES/FUNCTIONS########

# A point on the map
class Point:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0

# A region on the map which is used to define task MBR
class Region:
    def __init__(self):
        self.topLeft = Point()
        self.bottomRight = Point()

# Each new earthquake is stored as an EarthquakeEvent.
class EarthquakeEvent:
    def __init__(self):
        self.event_id = "" # event_id is the same as the key in EarthquakeEvents dictionary
        self.info = "" # info is the header of the file stored as a string
        self.intensity_xyz_URL = ""
        self.lat = 0.0 
        self.lon = 0.0
        self.mag = 0.0 # Magnitude
        self.max_mmi = 0.0 # Instrumental Intensity
        self.threshold_mmi = 0.0 
        self.MBR = Region() #Task region based on mmi threshold
    def populate(self,info): #Updates earthquakeEvent based on the self.info
        self.info = info
        MAG = 1; LAT = 2; LON = 3; 
        inf = self.info.split(" ")
        self.mag = float(inf[MAG])
        self.lat = float(inf[LAT])
        self.lon = float(inf[LON])
    def resetMBR(self): #Resets the MBR to values too large/too small to be a lat or lon
        self.MBR.topLeft.lon =  500.0
        self.MBR.topLeft.lat = -500.0
        self.MBR.bottomRight.lon = -500.0
        self.MBR.bottomRight.lat =  500.0

def get_xyz_link(event_id):
    return 'http://earthquake.usgs.gov/earthquakes/shakemap/global/shake/'+event_id+'/download/grid.xyz.zip'
    

###################core function####################

def readRSS (url):

    doc = feedparser.parse(url)

    
    for entry in doc.entries:
        temp_id = entry.id.rsplit(':',1)[1]
        if (temp_id in earthqaukeEvents):
            if (not silent):
                print ""+temp_id+" is already stored."
            pass
        else:
            if (not silent):
                print ""+temp_id+" added to events."
            earthqaukeEvents[temp_id] = EarthquakeEvent()
            earthqaukeEvents[temp_id].event_id = temp_id
            earthqaukeEvents[temp_id].threshold_mmi = THRESHOLD_MMI
            earthqaukeEvents[temp_id].intensity_xyz_URL = get_xyz_link(temp_id)
            
            try:
                response = urllib2.urlopen(earthqaukeEvents[temp_id].intensity_xyz_URL)
                zipHolder = cStringIO.StringIO(response.read())
                zipFile = zipfile.ZipFile (zipHolder)
                intensity_xyz_text = zipFile.read(zipFile.namelist()[0])
                lines =  intensity_xyz_text.split("\n")
                HEADER = 0
                earthqaukeEvents[temp_id].populate(lines[HEADER])
                earthqaukeEvents[temp_id].resetMBR()
                

                LON = 0; LAT =1; MMI = 2; 
                max_mmi = -1
                for line in lines [1:]:
                    data = line.split(" ")
                    if len(data) > MMI:
                        if float(data[MMI]) > earthqaukeEvents[temp_id].threshold_mmi:
                            if data[MMI] > max_mmi:
                                max_mmi = data[MMI]
                            if earthqaukeEvents[temp_id].MBR.topLeft.lon > float(data[LON]):
                                earthqaukeEvents[temp_id].MBR.topLeft.lon = float(data[LON])
                            if earthqaukeEvents[temp_id].MBR.topLeft.lat < float(data[LAT]):
                                earthqaukeEvents[temp_id].MBR.topLeft.lat = float(data[LAT])
                            if earthqaukeEvents[temp_id].MBR.bottomRight.lon < float(data[LON]):
                                earthqaukeEvents[temp_id].MBR.bottomRight.lon = float(data[LON])
                            if earthqaukeEvents[temp_id].MBR.bottomRight.lat > float(data[LAT]):
                                earthqaukeEvents[temp_id].MBR.bottomRight.lat = float(data[LAT])
                        else:
                            pass
                earthqaukeEvents[temp_id].max_mmi = max_mmi
                
                
                        
                
            except urllib2.HTTPError:
                print 'Error in reading intensity xyz file for '+temp_id+'. URL does not exists.'
    # To call readRSS every POLLING_INTERVAL seconds.
    threading.Timer(POLLING_INTERVAL, readRSS, [url]).start()

############## main() ####################

#sample URLs
significant_hour = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.atom"
significant_day = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.atom"
significant_month = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.atom"

# set to true to avoid prints
silent = False
# read RSS, and repeat every POLLING_INTERVAL seconds.
readRSS (significant_month)