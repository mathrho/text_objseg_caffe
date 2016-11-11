####################################################
videofiles = sorted(glob('/home/zhenyang/Workspace/data/OTB-100-othervideos/*'))
for videofile in videofiles:
    video = videofile.split('/')[-1]
    print(video)

    if video == 'ClifBar':
        continue

    # First, select query
    query_file = '../OTB100/OTB100Entities/' + video + '.xml'
    root = ET.parse( query_file ).getroot()
    # querier = prettify( querier )
    print(root[2][1].text)
    query = root[2][1].text

    with open('../OTB100/OTB100Queries/' + video + '.txt', 'w') as fp:
        fp.write(query+'\n')
    
