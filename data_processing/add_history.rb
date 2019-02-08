require 'csv'
Dir.glob("data_filled4/processed*.csv") do |f|
  puts f
  home_number = File.basename(f).split("_")[2]
  array=CSV.read("#{f}",:headers => true)
  # puts array[3][2]


  #localhour	use	temperature	cloud_cover	wind_speed	GH	is_weekday	month	hour	use_day	use_week	AC	DC
  #0          1        2          3           4       5      6          7     8      9       10     11  12
  line_number=0
  CSV.open("data_added4/added_hhdata_#{home_number}_4.csv","wb") do |csv|
  CSV.foreach("#{f}") do |row|
    if line_number==0
      csv << ["#{row[1]}","#{row[2]}","#{row[3]}","#{row[4]}","#{row[5]}","#{row[6]}","#{row[7]}","#{row[8]}","#{row[9]}","use_hour","use_week","ac","ac_hour","ac_week"]
    end
    #puts row
    if line_number>=169
      csv << ["#{row[1]}","#{row[2]}","#{row[3]}","#{row[4]}","#{row[5]}","#{row[6]}","#{row[7]}","#{row[8]}","#{row[9]}","#{array[line_number-2][2]}","#{array[line_number-169][2]}","#{row[10]}","#{array[line_number-2][10]}","#{array[line_number-169][10]}"]
    end
    line_number=line_number+1
  end
  end
end


# array=CSV.read("data/processed_hhdata_26.csv",:headers => true)
# # puts array[3][2]
#
# line_number=0
# CSV.open("data/added_hhdata_26.csv","wb") do |csv|
# CSV.foreach("data/processed_hhdata_26.csv") do |row|
#   if line_number==0
#     csv << ["#{row[0]}","#{row[1]}","#{row[2]}","#{row[3]}","#{row[4]}","#{row[5]}","#{row[6]}","#{row[7]}","GH_month","GH_week"]
#   end
#   #puts row
#   if line_number>=721
#     csv << ["#{row[0]}","#{row[1]}","#{row[2]}","#{row[3]}","#{row[4]}","#{row[5]}","#{row[6]}","#{row[7]}","#{array[line_number-721][6]}","#{array[line_number-169][6]}"]
#   end
#   line_number=line_number+1
# end
# end
