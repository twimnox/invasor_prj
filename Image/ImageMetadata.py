import exifread

# based on https://gist.github.com/erans/983821

class ImageMetadata(object):


    def __init__(self, variables):
        super(ImageMetadata, self).__init__()
        self.variables = variables
        f = open (self.variables.import_data_path, 'rb')
        self.tags = exifread.process_file(f)





    def _get_if_exist(self, data, key):
        if key in data:
            return data[key]

        return None


    def _convert_to_degress(self, value):
        """
        Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
        :param value:
        :type value: exifread.utils.Ratio
        :rtype: float
        """
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)


        return d + (m / 60.0) + (s / 3600.0)

    def get_focal_length(self):
        """
        Returns the focal lenght and focal length in 20mm film
        """
        exif_data = self.tags
        focal_length = self._get_if_exist(exif_data, 'EXIF FocalLength')
        focal_length_in_20mm = self._get_if_exist(exif_data, 'EXIF FocalLengthIn35mmFilm')

        numerator, denominador = str(focal_length).split('/')
        fl_float = int (numerator) / float (denominador)

        return fl_float, focal_length_in_20mm

    def get_exif_location(self):
        """
        Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)
        """
        lat = None
        lon = None
        alt = None
        exif_data = self.tags

        gps_latitude = self._get_if_exist(exif_data, 'GPS GPSLatitude')
        gps_latitude_ref = self._get_if_exist(exif_data, 'GPS GPSLatitudeRef')
        gps_longitude = self._get_if_exist(exif_data, 'GPS GPSLongitude')
        gps_longitude_ref = self._get_if_exist(exif_data, 'GPS GPSLongitudeRef')
        gps_altitude = self._get_if_exist(exif_data, 'GPS GPSAltitude')

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = self._convert_to_degress(gps_latitude)
            if gps_latitude_ref.values[0] != 'N':
                lat = 0 - lat

            lon = self._convert_to_degress(gps_longitude)
            if gps_longitude_ref.values[0] != 'E':
                lon = 0 - lon
        alt = str(gps_altitude)

        if ('/' in alt):
            numerator, denominador = str(alt).split('/')
            alt_float = int (numerator) / float (denominador)
            alt = float(alt_float)
        else:
            alt = float(alt)

        return lat, lon, alt


    def get_cm_per_1px_resolution(self, altitude, fl_mm):
        """
        Returns the cm/px resolution based on altitude and focal length.
        """
        fl_float = self.get_focal_length()[0]
        alti = self.get_exif_location()[2]

        ftprintx = ((alti/fl_float)*self.variables.CAMERA_SENSOR_WIDTH)*100
        ftprinty = ((alti/fl_float)*self.variables.CAMERA_SENSOR_HEIGTH)*100
        res = (ftprintx/float(self.variables.IMG_WIDTH) + ftprinty/float(self.variables.IMG_HEIGTH)) / float(2) #average measurements to reduce error

        return ftprintx, ftprinty, res





if __name__ == '__main__':
    import sys

    # SENSOR_HEIGTH = 4.62
    # SENSOR_WIDTH = 6.16

    #Test code (for just 1 image):
    im = ImageMetadata("empty")

    f = open (self.variables.import_data_path, 'rb')
    tags = exifread.process_file(f)
    lati, longi, alti = im.get_exif_location(tags)
    print lati, longi, alti

    alti = (alti)


    fl_float, fl_20 = im.get_focal_length(tags)
    print fl_float, 'mm' ,  fl_20, 'mm'

    # p1 = alti/10

    ftprintx = (alti/fl_float)*self.variables.CAMERA_SENSOR_WIDTH
    ftprinty = (alti/fl_float)*self.variables.CAMERA_SENSOR_HEIGTH
    print 'ft_x ', ftprintx, 'ft_y', ftprinty