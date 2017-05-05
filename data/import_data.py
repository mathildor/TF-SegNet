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

    for feature in dataLayer:
        geom = feature.GetGeometryRef()
        if not geom:
            continue
        geom.FlattenTo2D()
        id = feature.GetField("poly_id")
        objtype = feature.GetField("objtype")
        byggtyp = feature.GetField("byggtyp_nb")

        to_tuple = ( id, objtype, byggtyp, 'SRID=25832;' + geom.ExportToWkt())

        to_cur.execute("""INSERT into ar_bygg (id, objtype,  byggtyp, geom)
                        SELECT %s, %s, %s, ST_GeometryFromText(%s);""",
                    to_tuple)

    to_conn.commit()
    dataSource.Destroy()

# to_conn = psycopg2.connect("host=localhost port=5433 dbname=ar-bygg-ostfold user=postgres password=24Pils")
to_conn = psycopg2.connect("host=localhost port=5432 dbname=Fet user=postgres password=123")
to_cur = to_conn.cursor()

# by_filebase = "./ar5_bygg_01/32_FKB_{0}_Bygning/32_{0}bygning_flate.shp"
by_filebase = "../../aerial_datasets/FKB_area_for_IR_and_RGB/32_FKB_{0}_Bygning/32_{0}bygning_flate.shp"

#Kommune nummer liste - spesifisert i mappenavnet
k_nr_list = [ #For FKB_area_for_IR_ad_RGB
    #"0228" #ralingen
    #"0230" #lorenskog
    #"0231" #skedsmo
    #"0233" #nittedal
    #"1003" #farsund
    #"0226"  #sorum
    "0227"  #fet
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

for k_nr in k_nr_list:
    # ar_file = ar_filebase.format(k_nr)
    by_file = by_filebase.format(k_nr)
    print('by_file')
    print(by_file)

    # importAr(to_cur, ar_file)
    importBygg(to_cur, by_file)
