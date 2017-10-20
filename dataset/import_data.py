from __future__ import division
from __future__ import print_function

from osgeo import ogr
import psycopg2



def importAr(to_cur, filename):
    print("Importing:", filename)
    dataSource = ogr.Open(filename)
    dataLayer = dataSource.GetLayer(0)

    for feature in dataLayer:
        geom = feature.GetGeometryRef().ExportToWkt()
        id = feature.GetField("poly_id")
        objtype = feature.GetField("objtype")
        artype = feature.GetField("artype")
        arskogbon = feature.GetField("arskogbon")
        artreslag = feature.GetField("artreslag")
        argrunnf = feature.GetField("argrunnf")

        to_tuple = ( id, objtype, artype, arskogbon, artreslag, argrunnf, geom)
        to_cur.execute("""INSERT into ar_bygg (id, objtype, artype, arskogbon, artreslag, argrunnf, geom)
                        SELECT %s, %s, %s, %s, %s, %s, ST_GeometryFromText(%s);""",
                   to_tuple)

    to_conn.commit()
    dataSource.Destroy()

def importBygg(to_cur, filename):
    print("Importing:", filename)
    dataSource = ogr.Open(filename)
    dataLayer = dataSource.GetLayer(0)
    print("dataLayer")
    print(dataLayer)

    #Genererer id'er fortloopende
    for id, feature in enumerate(dataLayer):
        geom = feature.GetGeometryRef()
        if not geom:
            continue
        geom.FlattenTo2D()
        print("Feature")
        print(feature)

        for i in range(1, feature.GetFieldCount()):
            field = feature.GetDefnRef().GetFieldDefn(i).GetName()
            if( i == 4):
                continue
            #print(field.encode('utf-8'))

        byggtyp = feature.GetField("BYGGTYP_NB")
        #poly_id = feature.GetField("LOKALID   ")
        objtype = feature.GetField("OBJTYPE")

        to_tuple = (id, objtype, byggtyp, 'SRID=25832;' + geom.ExportToWkt())

        to_cur.execute("""INSERT into ar_bygg (id, objtype,  byggtyp, geom)
                        SELECT %s, %s, %s, ST_GeometryFromText(%s);""",
                    to_tuple)

    to_conn.commit()
    dataSource.Destroy()

# to_conn = psycopg2.connect("host=localhost port=5433 dbname=ar-bygg-ostfold user=postgres password=24Pils")
to_conn = psycopg2.connect("host=localhost port=5432 dbname=Asker user=postgres password=1234")
to_cur = to_conn.cursor()

# by_filebase = "./ar5_bygg_01/32_FKB_{0}_Bygning/32_{0}bygning_flate.shp"
#by_filebase = "../../fkb-data/32_FKB_{0}_Bygning/32_{0}bygning_flate.shp"
by_filebase = "../../fkb-data/Asker-shape/Bygning_polygon.shp"

#Kommune nummer liste - spesifisert i mappenavnet
k_nr_list = [ #For FKB_area_for_IR_ad_RGB
    #"0228" #ralingen
    #"0230" #lorenskog
    #"0231" #skedsmo
    #"0233" #nittedal
    #"1003" #farsund
    #"0226"  #sorum
    #"0227"  #fet
    #"1601" #Trondheim
    "0220" #Asker
]
#k_nr_list = [ #For IR_area_fkb mappen
    # "0211"
    # "0213",
    # "0214",
    # "0215",
    # "0216",
    # "0217",
    # "0219",
    # "0220",
    #"0426",
    #"0427"
#]

# k_nr_list = [ #For ar5_bygg_01 mappen
#     "0101",
#     "0104",
#     "0105",
#     "0106",
#     "0111",
#     "0118",
#     "0119",
#     "0121",
#     "0122",
#     "0123",
#     "0124",
#     "0125",
#     "0127",
#     "0128",
#     "0135",
#     "0136",
#     "0137",
#     "0138",
# ]

print('by_file')
print(by_filebase)

# importAr(to_cur, ar_file)
importBygg(to_cur, by_filebase)

''' for k_nr in k_nr_list:
    # ar_file = ar_filebase.format(k_nr)
    #by_file = by_filebase.format(k_nr)
    print('by_file')
    print(by_filebase)

    # importAr(to_cur, ar_file)
    importBygg(to_cur, by_filebase) '''
