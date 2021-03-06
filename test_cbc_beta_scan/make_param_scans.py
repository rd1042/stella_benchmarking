"""Code to generate scans in beta' for both stella (running on MARCONI)
and GS2 (running on VIKING) """

import numpy as np
import re
import sys
import os
import shutil

## Define folder names here
# stella
# stella_fapar0_fbpar1_me1_folder = "stella_fapar0_fbpar1_me1_beta_scan"
# stella_fapar1_fbpar0_me1_folder = "stella_fapar1_fbpar0_me1_beta_scan"
# stella_fapar1_fbpar1_me1_folder = "stella_fapar1_fbpar1_me1_beta_scan"
stella_fapar0_fbpar1_folder = "stella_fapar0_fbpar1_beta_scan"
stella_fapar1_fbpar0_folder = "stella_fapar1_fbpar0_beta_scan"
stella_fapar1_fbpar1_folder = "stella_fapar1_fbpar1_beta_scan"


# gs2
gs2_fapar0_fbpar1_me1_folder = "gs2_fapar0_fbpar1_me1_beta_scan"
gs2_fapar1_fbpar0_me1_folder = "gs2_fapar1_fbpar0_me1_beta_scan"
gs2_fapar1_fbpar1_me1_folder = "gs2_fapar1_fbpar1_me1_beta_scan"

def construct_beta_scan(folder_name, value_list, sim_type):
    template_name = folder_name + "/template_input_file"
    template_runscript = folder_name + "/template_run_script.sh"

    # Check folder exists
    if not os.path.isdir(folder_name):
        sys.exit("folder " + folder_name  + " does not exist!")

    # Check if the template file exists
    if not os.path.isfile(template_name):
        sys.exit("template file does not exist!")


    make_input_files_1d(template_name, template_runscript, 'beta', value_list, 'beta', folder=folder_name)
    shutil.copy('../templates_and_scripts/submit_jobs.sh', folder_name); os.chmod((folder_name + '/submit_jobs.sh'), 0o777)
    print("folder contents: " + folder_name)
    os.system('ls -ltr ' + folder_name)
    return


def make_input_files_1d(template, template_runscript, parameter, value_list, filename_prefix, folder="."):
  """Constructs many input files, changing a single parameter to values based
  a list"""

  for value in value_list:
      make_input_file(template, template_runscript, [parameter], [value], filename_prefix=filename_prefix,
                        folder=folder)

  return

def make_input_file(template, template_runscript, parameters, values, beware_integers=False, filename_prefix="",
                    filename=False, folder="."):
    """Constructs an input file based on the template, changing an arbitrary number of
    parameters."""

    ## Check if same number of parameters as values
    if len(parameters) != len(values):
        print("Error! len(parameters), len(values) = ", len(parameters), len(values))
        sys.exit()
    template_file = open(template, 'r+')
    filetext = template_file.read()
    template_file.close()
    values_str = ""
    for i, parameter in enumerate(parameters):
      value = values[i]
      if beware_integers:
          if isinstance(value, int):
              value_str = str(value)
          elif isinstance(value, float):
            value_str = ('%.4f' % value)
          else:
              print("Error! Type not recognised. value = ", value)
      else:
          value_str = ('%.5f' % value)

      values_str += ("_" + value_str)

      if type(parameter) is str:
          search_string = parameter + " = .*"
          replace_string = parameter + " = " + value_str
          filetext = re.sub(search_string, replace_string, filetext)
      elif type(parameter) is list:
          for each_parameter in parameter:
              search_string = each_parameter + " = .*"
              replace_string = each_parameter + " = " + value_str
              filetext = re.sub(search_string, replace_string, filetext)
      else:
          print("type(parameter) not recognised! Aborting" )
          print("type(parameter) = ", type(parameter))
          sys.exit()

    if filename != False:
        new_filename = filename
    else:
        #print("folder = ", folder)
        infile_shortname = filename_prefix + values_str + ".in"
        new_filename = folder +  "/" + infile_shortname

    #print("new_filename = ", new_filename)
    #sys.exit()
    new_file = open(new_filename, "w+")
    new_file.write(filetext)
    new_file.close()

    ## Open and modify run script.
    template_file = open(template_runscript, 'r+')
    filetext = template_file.read()
    template_file.close()
    filetext = re.sub("input.in", infile_shortname, filetext)
    new_runscript_name = folder + "/run_beta" + values_str + ".sh"
    new_file = open(new_runscript_name, "w+")
    new_file.write(filetext)
    new_file.close()
    return


def make_beta_scans_for_me1():
    """Construct simulations scanning beta for electron mass=1.
    The test cases are:
    (1) fapar=0, fbpar=1
    (2) fapar=1, fbpar=0
    (3) fapar=1, fbpar=1 """


    beta_vals = np.linspace(0, 0.04, 21)

    ##
    # construct_beta_scan(stella_fapar1_fbpar0_me1_folder, beta_vals, "stella")
    # construct_beta_scan(stella_fapar1_fbpar1_me1_folder, beta_vals, "stella")
    # construct_beta_scan(stella_fapar0_fbpar1_me1_folder, beta_vals, "stella")
    construct_beta_scan(gs2_fapar1_fbpar0_me1_folder, beta_vals, "stella")
    construct_beta_scan(gs2_fapar0_fbpar1_me1_folder, beta_vals, "stella")
    construct_beta_scan(gs2_fapar1_fbpar1_me1_folder, beta_vals, "stella")

    return

def make_beta_scans_for_stella_2spec():
    """Construct simulations scanning beta for stella; 2 species, normal electron mass.
    The test cases are:
    (1) fapar=0, fbpar=1
    (2) fapar=1, fbpar=0
    (3) fapar=1, fbpar=1 """


    beta_vals = np.linspace(0, 0.04, 21)

    ##
    # construct_beta_scan(stella_fapar1_fbpar0_me1_folder, beta_vals, "stella")
    # construct_beta_scan(stella_fapar1_fbpar1_me1_folder, beta_vals, "stella")
    # construct_beta_scan(stella_fapar0_fbpar1_me1_folder, beta_vals, "stella")
    # construct_beta_scan(stella_fapar0_fbpar1_folder, beta_vals, "stella")
    # construct_beta_scan(stella_fapar1_fbpar0_folder, beta_vals, "stella")
    construct_beta_scan(stella_fapar1_fbpar1_folder, beta_vals, "stella")

    return



if __name__ == "__main__":
    print("Hello world")
    # make_beta_scans_for_me1()
    make_beta_scans_for_stella_2spec()
