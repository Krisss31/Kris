import streamlit as st 
import numpy as np 
import pandas as pd
from joblib import load


model = load('Kris/Number_of_Pumps_XGB.joblib')
data = pd.read_csv('colonne_camion.csv')

Property_type = ['Park ', 'Purpose Built Flats/Maisonettes - Up to 3 storeys ',
       'Tree scrub ', 'Hospital ', 'House - single occupancy ',
       'Purpose Built Flats/Maisonettes - 4 to 9 storeys ',
       'Common external bin storage area', 'Student Hall of Residence ',
       'Self contained Sheltered Housing ',
       'Converted Flat/Maisonettes - 3 or more storeys', 'Loose refuse ',
       'Hedge', 'Lorry/HGV ', 'Hostel (e.g. for homeless people) ',
       'Motorcycle ', 'Converted office ',
       'Other medical establishment (including surgery) ',
       'Converted Flat/Maisonette - Up to 2 storeys ', 'Car ',
       'Retirement/Old Persons Home', 'College/University ',
       'Large refuse/rubbish container (eg skip/ paladin)',
       'Towing caravan (not on tow or on site)', 'Infant/Primary school',
       'Single shop ',
       'House in Multiple Occupation - 3 or more storeys (not known if licensed)',
       'Small refuse/rubbish container', 'Secondary school',
       'Hotel/motel ', 'Bus/coach ', 'Warehouse ',
       'Purpose built office ', 'Medical/health centre', 'Telephone box ',
       'Fence', 'Leisure Centre ', 'Restaurant/cafe', 'Library ',
       'Laundrette ',
       'Purpose Built Flats/Maisonettes - 10 or more storeys ',
       'Petrol station ',
       'Licensed House in Multiple Occupation - Up to 2 storeys ',
       'Museum ', 'Train station - platform (below ground) ',
       'Church/Chapel ', 'Private garage ', 'Food and drink processing',
       'Youth hostel ', 'Cinema ', 'Police station ',
       'Refuse/rubbish tip ', 'Multiple Vehicles ',
       'Other entertainment venue ', 'Water works ', 'Golf clubhouse',
       'Manufacturing assembly plant', 'Shelter ', 'Cables ',
       'Engineering manufacturing plant', 'Indoor Market ', 'Gym ',
       'Domestic garden (vegetation not equipment) ', 'Bakery',
       'DIY Warehouse ', 'Van ', 'Other retail warehouse ',
       'House in Multiple Occupation - Up to 2 storeys (not known if licensed) ',
       'Woodland/forest - conifers/softwood ', 'Large supermarket ',
       'Animal harm outdoors', 'Pub/wine bar/bar ',
       'Other outdoor structures ', 'Laboratory/research Establishment ',
       'Other education establishment', 'Railings',
       'Bus/coach station/garage ', 'Bungalow - single occupancy ',
       'Boarding House/B&B for homeless/asylum seekers ',
       'Community centre/Hall ', 'Private Summer house ',
       'Takeaway/ fast food ', 'Factory ', 'Telephone exchange',
       'Roadside furniture (eg lamp posts/ road signs/ telegraph poles/ speed cameras)',
       'Local Government Office', "Nurses'/Doctors' accommodation ",
       'Department Store ', 'Shopping Centre ', 'Road surface/pavement ',
       'Nursing/Care Home/Hospice', 'Doctors surgery',
       'Other road vehicle', 'TV/film/music/art studio ',
       'Other office/call centre type building', 'Vehicle sales building',
       'Other private non-residential building ', 'Cricket ground ',
       'Young offenders unit ', 'Art Gallery ',
       'Other building/use not known ', 'Veterinary surgery',
       'Other retail  ', 'Other cultural venue ',
       'Other Residential Home ', 'Bingo Hall ', 'Other public building ',
       'Pre School/nursery ',
       'Playground/Recreation area (not equipment)', 'Hairdresser ',
       'Licensed House in Multiple Occupation - 3 or more storeys ',
       'Club/night club ', 'Wheelie bin (domestic size)', 'Scrub land ',
       'Town Hall ', 'Airport building (not terminal or hangar) ',
       'Railway building - other', 'Electricity power station ',
       'Roadside vegetation ', 'Other car park structure',
       'Other holiday residence (cottage/ flat/ chalet) ',
       'Central Government Office', 'Monastery/convent ',
       'Other outdoor equipment/machinery ', 'Vehicle Repair Workshop',
       'Train station - elsewhere ', 'Public toilets ',
       'Private Garden Shed ', 'Multi-Storey car park',
       'Cycle path/public footpath/bridleway ', 'Recycling plant',
       'Bank/Building Society ', 'Dental surgery',
       'Canal/riverbank vegetation ', 'Temporary office (eg portacabin) ',
       'Passenger plane ',
       'Unlicensed House in Multiple Occupation - Up to 2 storeys ',
       'Outdoor storage ', 'Call Centre ', 'Football stadium ',
       'River/canal ', 'Law Courts ', 'Lake/pond/reservoir ', 'Barge ',
       'Sports pavilion/shower block/changing facility ', 'Fire station ',
       'Grassland/ pasture/ grazing etc ', 'Sports/Social club ',
       'Cathedral ', 'Other public utility works',
       'Other industrial manufacturing facility', 'Swimming Pool ',
       'Exhibition Centre ', 'Military/barracks ', 'Pipe or drain ',
       'Other Dwelling ', 'Electrical warehouse ', 'Estate Agent ',
       'Theatre ', 'Cemetery ', 'Mosque ', 'Other outdoor location ',
       'Other bulk storage', 'Prison ', 'Sewage works ', 'Bicycle',
       'Concert Hall ', 'Chemical plant', 'Railway trackside vegetation ',
       'Train station - platform (at ground level or elevated) ',
       'Road Tanker ', 'Other Religious Use Building', 'Bridge',
       'Distillery plant', 'Other indoor sporting venue ', 'Mill ',
       'Wasteland ', 'Animal boarding/breeding establishment - dogs',
       'Passenger Train (national rail network) ', 'Garden equipment ',
       'Post box ', 'Athletics Stadium ', 'Barn ', 'Ice rink ',
       'Human harm outdoors', 'Heathland ', 'Printing works', 'Railway ',
       'Underground car park', 'Recycling collection point/ bottle bank',
       'Airport - terminal ',
       'Other animal boarding/breeding establishment',
       'Nurseries/ market garden ', 'Day care/Drop in centre',
       'Health Centre ', 'Stacked/baled crop ', 'Health spa/farm',
       'Train station - concourse ', 'Furniture warehouse ',
       'Caravan/Mobile home (permanent dwelling)', 'Tunnel/ subway ',
       'Other industrial processing plant', 'Boarding House/B&B other ',
       'Train on Tube network', 'Underground train : Other system ',
       'Post office (purpose built) ',
       'Trailer (not attached to tractor unit)', 'Conference Centre ',
       'Other transport building ', 'Sports Hall ', 'Synagogue ',
       'Unlicensed House in Multiple Occupation - 3 or more storeys ',
       'Stately Home (part not open to public)', 'Trains - engine shed ',
       'Casino ', 'Golf course (not building on course)', 'Kiosk ',
       'Private greenhouse ', 'Boarding School accommodation ',
       'Bulk waste storage', 'Ambulance station ', 'Tennis Courts ',
       'Airport - hangar ', 'Woodland/forest - broadleaf/hardwood ',
       'Royal Palace (part not open to public)', 'Zoo ', 'Travel Agent ',
       'Straw/stubble burning ', 'Other vessel ',
       'Animal boarding/breeding establishment - cats', 'Motor Home ',
       'Landfill site ', "Children's Home", 'Tractor Shed ',
       'Ministry of Defence office', 'Bulk oil storage',
       'Other tent/marquee ', 'Other outdoor sporting venue ', 'Temple ',
       'Rugby Stadium ', 'Houseboat (permanent dwelling) ', 'Minibus ',
       'Mine or quarry (not above ground building)',
       'Post office (within other shop/premises) ', 'Oil refinery ',
       'Barbecue', 'Naval vessel ', 'Camping tent ', 'Caravan on tow ',
       'Gas works ', 'Other aircraft', 'Freight Train ',
       'Agricultural vehicle', 'Bulk gas storage',
       'Other agricultural building ', 'Indoor stadium ', 'Beach ',
       'False Alarm - Property not found', 'Standing crop ',
       'Animal products processing plant', 'Helicopter ',
       'Railway goods yard ', 'Greyhound stadium ', 'Airfield/runway ',
       'Fishing boat ', 'Docks ', 'Ferry terminal ', 'Motor yacht ',
       'Castle (part not open to public)',
       'Greenhouse (commercial) glass ', 'Large passenger vessel ',
       'Sea ', 'Towing caravan/Camper van on site', 'Light aircraft ',
       'Agricultural equipment ', 'Theme Park ',
       'Bulk hazardous materials storage',
       'Mine or quarry building above ground', 'Airport - fuel storage ',
       'Tram ', 'Freight plane ',
       'Intensive Farming Sheds (chickens/ pigs etc) ',
       'Other merchant vessel ']

Property_category = ['Outdoor',
                     'Dwelling', 
                     'Non Residential', 
                     'Outdoor Structure',
                     'Other Residential',
                     'Road Vehicle',
                     'Aircraft',
                     'Boat',
                     'Rail Vehicle']

AddressQualifier = ['Open land/water - nearest gazetteer location',
       'Correct incident location', 'On land associated with building',
       'In street outside gazetteer location', 'Within same building',
       'In street close to gazetteer location',
       'On motorway / elevated road',
       'Nearby address - street not listed in gazetteer',
       'Nearby address - no building in street',
       'In street remote from gazetteer location',
       'Railway land or rolling stock']

IncidentCategory = ['Effecting entry/exit', 'Secondary Fire', 'AFA', 'Lift Release','Flooding', 'Primary Fire', 'RTC', 'Advice Only',
       'Other rescue/release of persons', 'Medical Incident',
       'Assist other agencies', 'Evacuation (no fire)',
       'Spills and Leaks (not RTC)', 'Chimney Fire',
       'Hazardous Materials incident', 'Removal of objects from people',
       'Making Safe (not RTC)', 'Suicide/attempts',
       'Animal assistance incidents', 'Other Transport incident',
       'Stand By', 'Rescue or evacuation from water', 'Late Call',
       'Medical Incident - Co-responder', 'Water provision',
       'Use of Special Operations Room']

IncidentType = ['Domestic Incidents', 'Fire', 'Major Environmental Disasters',
       'Local Emergencies', 'Prior Arrangement',
       'Use of Special Operations Room']
    
if st.sidebar.checkbox("Do you want to choose the Property_type", False):
   features_options = Property_type
   features = st.multiselect("Please choose the features including target variable that go into the model", features_options)
   data['features'] = 1
   st.write(data)

