import specpy
import numpy
import os
from PIL import Image
import scipy.ndimage as ndimage
import configparser

# work around for the fact that sometimes I duplicated the STED channels during imaging and they are saved in the measurement but ImSpector forgot the name.
EMPTY = " "
STED = "STED"
CH2 = "Ch2 {2}"  # auf der alten Waldweg software hab ich das meist als af594 Kanal gehabt
CH4 = "Ch4 {2}"
# falls der stack dann immer noch größer 2 ist wird der Rest einfach gepoppt und zwar:
# in der read_stack_from_imspector_measurement function in der Liste wanted_stack
#TODO: make work for subdirectories, make continue loop even if one stack doesn't work

def main():
    # root_path = r"P:\Private\practice\imaging\IF\selected\IFs_Freiburg_replicates\01_U2OS_DKO_plus_Bax_wt" # r stands for raw. So that it doesn't read the stupid windows way of filepaths with backslashes wrong.
    root_path = get_root_path()
    result_path = os.path.join(root_path, 'extracted_tifs_from_msr')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    filenames = list(os.listdir(root_path))
    for filename in filenames:  # ich erstelle eine Liste mit den Filenames in dem Ordner
        if filename.endswith(".msr"):  # wenn die Endung .msr ist, dann mach was damit, nämlich:
    # ATTENTION: the following file name is too long and it will give an error like: FileNotFoundError: [Errno 2] No such file or directory:
    # all of these need to be shortened.
    # filename = "IF41.2_spl6_U2OS-DKO_pcDNA-Bax-BH3i_6hEx_14hAct_cytC-M394-AF488_Tom20-M440-AF594_Bax-M482-SR_cl2_Bax-blobs_cytoBax_cytC-release.msr"
            print(filename)
            file_path = os.path.join(root_path, filename)
            stacks = read_stack_from_imspector_measurement(file_path)
            images, stack_names = make_image_from_imspector_stack(stacks)
            if len(images) != 2:
                print('Problem: {} ImSpector stacks, need two.'.format(len(images)))
                return
            for i in range(len(images)):
                image = images[i]
                stackname = stack_names[i]
                extra_factor = determine_extra_factor(i)

                denoised_data = gaussian_blur(image)
                enhanced_contrast = enhance_contrast(denoised_data, extra_factor)

                # save the original
                save_array_with_pillow(image, result_path, filename, stackname + str(i))
                # save the denoised and contrast enhanced
                save_array_with_pillow(enhanced_contrast, result_path, filename, stackname + str(i) + "contr_enh")


def get_root_path():
    """
    Retrieves the root path
    """
    config = configparser.ConfigParser()
    config.read('extract_stacks_from_msr_and_save_as_tif.ini')  # wenn es ein Verzeichnis höher liegen würde wäre es ("../hbujbk")
    return config['general']['root-path']


def read_stack_from_imspector_measurement(file_path):
    """
    Lädt die Imspector Messung und findet die Kanäle (=stacks) die wir mit namepart sepzifizieren.

    :param file_path: Pfad der Imspector Messung.
    :param name_part: Teil des Stacknamens
    :return: Alles Kanäla (=Stacks), die so heißen
    """

    # File lesen
    measurement = specpy.File(file_path, specpy.File.Read)


    # lese alle stacks in eine liste
    all_stacks = []  #empty list
    number_stacks = measurement.number_of_stacks()  # returns the number of stacks in the measurement
    for i in range(number_stacks):
        stack = measurement.read(i)  #ein Kanal wird in Imspector als stack bezeichnet!
        all_stacks.append(stack)
    print('The measurement contains {} channels.'.format(len(all_stacks)))  # gibt mir aus wie viele Kanäle die Messung hat

    # finde alle stacks, deren name entweder das Wort STED enthält, oder welcher keine spaces (EMPTY) enthält, das kann dann nur der leere (=duplizierte) sein.
    # die CONSTANTS dafür sind oben vor der main() definiert = workaround für channel duplication und bescheuerte ImSpector Benennungen..
    wanted_stack_s = [stack for stack in all_stacks if EMPTY not in stack.name() or STED in stack.name() or CH2 in stack.name() or CH4 in stack.name()]  # list comprehension(?)  #stack.name() ist von specpy
    print('The measurement contains {} STED channels.'.format(len(wanted_stack_s)))

    # if we get more than 2 stacks (one AF594 and one STAR RED) then it's most likely duplicates and we will just remove them from the list
    if len(wanted_stack_s) > 2:
        wanted_stack_s = wanted_stack_s[:2]

    #OR: wanted_stack_s[:min(len(wanted_stack_s), 2)]  # this gives back the wanted_stack_s list from the start up until before 2 (so 0 and 1).
    #However, if the stack is smaller then it should only return up until the length of the list, otherwise we will get an index error.
    #that's why we need the minimum of the two values. Either length of list or 2.


    return wanted_stack_s


def make_image_from_imspector_stack(wanted_stack_s):
    """

    :param wanted_stack_s: die ausgewählten Bilder einer Messung
    :return: Numpy array davon
    """
    stack_size = len(wanted_stack_s)
    images = []  # die leere Liste wo ich meine Ergebnissbilder reinspeichere
    stacknames = []  #TODO. make dictionary??
    for i in range(stack_size):  # TODO: diese for loop in die main und ab data= erst hier lassen
        wanted_stack = wanted_stack_s[i]  # muss ein Element aus der Liste rausfangen, damit ich es in ein numpy array umwandeln kann.
        stack_name = wanted_stack_s[i].name()
        data = wanted_stack.data()  # returns the data of the stack as a NumPy array

        # Dimensionnen von Imspector aus sind [1,1,Ny,Nx]
        size = data.shape  # The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n
        # print('The numpy array of the current {} channel has the following dimensions: {}'.format(NAME_PART, size))

        # wir wollen aber [Nx, Ny]
            # 1) reduce to [Ny, Nx]
        data = numpy.reshape(data, size[2:])

            # 2) transponieren [Nx, Ny]
        data = numpy.transpose(data)

            # 3) just to visualize the dimensions again
        size = data.shape
        print('The numpy array of the {} channel has the following dimensions: {}'.format(wanted_stack.name(), size))

        images.append(data)
        stacknames.append(stack_name)

    return images, stacknames


def determine_extra_factor(i):
    '''
    the stack we extract from the imspector measurement should only have 2 objects, and the first one is AF594 and the second one star red.
    As long as Bax has been imaged in Starred, then these factors will suit more or less.
    '''
    #activate thisTODO if needed: Achtung es ist grade verdreht, weil ich ein sample set bearbeitet habe, wo bax im 594er Kanal liegt
    if i == 0:
        extra_factor = 2.5  # applied to AF594 channel (hier für mito contrast)
    elif i == 1:
        extra_factor = 5  # applied to starred channel (hier für Bax contrast)
    print(extra_factor)
    return extra_factor


def gaussian_blur(numpy_array):
    #Gaussian blur with scipy package
    denoised_data = ndimage.gaussian_filter(numpy_array, sigma=2)
    return denoised_data


def enhance_contrast(numpy_array, random_extra_factor):
    # Enhance contrast by stretching the histogram over the full range of the grayvalues
    minimum_gray = numpy.amin(numpy_array)
    maximum_gray = numpy.amax(numpy_array)
    mean_gray = numpy.mean(numpy_array)
    print("And the following greyvalue range: {} - {}, with a mean of: {}.".format(str(minimum_gray), str(maximum_gray), str(mean_gray)))
    factor = 255/maximum_gray
    # mean_factor = 127.5 / maximum_gray  # TODO: pick out all the pixels that are not 0 and calculate the mean
    print(factor)
    enhanced_contrast = numpy_array * factor * random_extra_factor  # depends on the position of the measurement in the stack
    thresh = 255
    super_threshold_indices = enhanced_contrast > thresh  # ich suche mir die Indices im Array, die über dem Threshold liegen
    enhanced_contrast[super_threshold_indices] = 255  # und setze die Intensitäten an diesen Stellen auf 255
    return enhanced_contrast


def save_array_with_pillow(image, result_path, filename, stackname):
    # I need to change the type of the numpy array to unsigned integer, otherwise can't be saved as tiff.
    # unit8 = Unsigned integer (0 to 255); unit32 = Unsigned integer (0 to 4294967295)
    eight_bit_array = image.astype(numpy.uint8)
    output_file = os.path.join(result_path, filename[:-4] + stackname + '.jpg')
    # print("wanted stack : {}".format(stackname)
    img = Image.fromarray(eight_bit_array)
    # print("I will save now")
    img.save(output_file, format='jpeg')  #TODO: save metadata (specifically pixel size for FIJI)


if __name__ == '__main__':
    main()
