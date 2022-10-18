
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel
storage_url = "D:/University/Masters/Dissertation/test-results/"

# save all SNN data into a .pkl file encase of future needs
def save_results(layer, file):
    date = save_date()
    layer.write_data(storage_url+''+file+'-'+date+'.pkl', 'spikes')
# returns current date and time, avoids constant repeating of code
def save_date():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    date = dt_string.replace('/', '-')
    date = date.replace(':', '-')
    date = date.replace(' ', '-')
    return date
# saves routes found from path planner to avoid new route on every run and decrease runtime
def log_tree(path ,title, name):
    datet = save_date()
    f = open("log_data/ " +title +"- " +str(path.initial_pos ) +"- " +str(path.goal ) +"-RRT- " +datet +".txt", "w+")
    f.write("Node length: \r\n")
    f.write("%s\n" % len(path.nodes_list))
    f.write("Node List coordinates: \r\n")
    for line in path.nodes_list:
        f.write("%s\n" % line)
    f.write("\nPath length: \r\n")
    f.write("%s\n" % len(path.path_nodes))
    f.write("\nPath Nodes coordinates: \r\n")
    f.write("%s\n" % path.path_nodes)
    f.close()

    f = open("map_paths/" + name + "-" + datet + ".csv",
             "w+")
    for line in path.path_nodes:
        f.write("{' "+ str(line['name']) +"', " + str(line['coord'][0]) + ", " + str(line['coord'][1]) + ", '" + str(
            line['parent']) + "',"
                              "" + str(line['degrees']) + "," + str(line['cost']) + "," + str(line['time']) + "}")
        if str(line['name']) != "n0":
            f.write(", \n")
    f.close()
# Store simulation data for SNN training
def save_location_training_data(log, title):
    datet = save_date()
    f = open("log_data/training_data/" + title + "-" +datet+".py", "w+")
    f.write("velocity=[")
    for velocity in log:
        f.write("["+str(velocity[0])+"," +str(velocity[2])+ "], \n")
    f.write("]\n")
    f.write("cur_pos=[")
    for cur_pos in log:
        f.write("["+str(cur_pos[0])+"," +str(cur_pos[3])+ "], \n")
    f.write("]\n")
    f.write("landmarks=[")
    for landmarks in log:
        f.write("["+str(landmarks[0])+", [")
        for L in landmarks[1]:
            f.write(str(L)+", ")
        f.write("]],\n")
    f.write("]")
    f.close()
#plot snn results
def plot_result(population, title, snn_name = ""):
    population.record('spikes')
    population[0:2].record('v')#, 'gsyn_exc'))

    data = population.get_data().segments[0]

    vm = data.filter(name="v")[0]
    gsyn = data.filter(name="gsyn_exc")[0]
    date = save_date()
    Figure(
        Panel(vm, ylabel="Membrane potential (mV)"),
        Panel(gsyn, ylabel="Synaptic conductance (uS)"),
        Panel(data.spiketrains, xlabel="Time (ms)", xticks=True)
    )#.save("plots/"+snn_name+""+date+"-simulation_results"+title+".png")
#plot snn spikes
def plot_spiketrains(segment, title, snn_name = ""):
    c = 0
    date = save_date()
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel(segment.name)
        plt.title("Spiketrain")
        plt.savefig(storage_url+"plots/"+snn_name+""+date+"-"+title+"_"+str(c)+"_spike_train.jpg")
        c+=1
        plt.setp(plt.gca().get_xticklabels(), visible=False)
#plot snn signals
def plot_signal(signal, index, title, snn_name = "", colour='b'):
    date = save_date()
    label = "Neuron %d" % signal.annotations['source_ids'][index]
    plt.plot(signal.times, signal[:, index], colour, label=label)
    plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.title("Signal")
    plt.savefig(storage_url+"plots/"+snn_name+""+date+"-"+title+".jpg")
    plt.legend()
#plot snn grouped results
def plot_results3(p_in, p_out, title, snn_name):
    spikes_in = p_in.get_data()
    data_out = p_out.get_data()

    fig_settings = {
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.labelsize': 'small',
        'legend.fontsize': 'small',
        'font.size': 8
    }
    plt.rcParams.update(fig_settings)
    plt.figure(1, figsize=(6, 8))

    n_panels = sum(a.shape[1] for a in data_out.segments[0].analogsignals) + 2
    plt.subplot(n_panels, 1, 1)
    plot_spiketrains(spikes_in.segments[0], title, snn_name)
    plt.subplot(n_panels, 1, 2)
    plot_spiketrains(data_out.segments[0],  title, snn_name)
    panel = 3
    for array in data_out.segments[0].analogsignals:
        for i in range(array.shape[1]):
            plt.subplot(n_panels, 1, panel)
            plot_signal(array, i, title, snn_name)
            panel += 1
    plt.xlabel("time (%s)" % array.times.units._dimensionality.string)
    plt.setp(plt.gca().get_xticklabels(), visible=True)
    date = save_date()
    #plt.savefig(storage_url+"plots/"+snn_name+""+date+"-"+title+".jpg")

    plt.show()
    plot_spiketrains(spikes_in.segments[0], title, snn_name)
    plt.show()
    plot_spiketrains(data_out.segments[0], title, snn_name)
    plt.show()

    for array in data_out.segments[0].analogsignals:
        for i in range(array.shape[1]):
            plot_signal(array, i, title, snn_name)
            panel += 1
    plt.xlabel("time (%s)" % array.times.units._dimensionality.string)
    plt.setp(plt.gca().get_xticklabels(), visible=True)
    plt.title("Signal")
    date = save_date()
    plt.savefig(storage_url+"plots/"+snn_name+""+date+"-"+title+"_signal.jpg")
    plt.show()
#plot snnn localisation results
def plot_positions(populationx, populationy, title, snn_name = ""):

    x_spikes = populationx.get_data('spikes')
    y_spikes = populationy.get_data('spikes')

    x_spikes = x_spikes.segments[0]
    x_plots = []
    y_spikes = y_spikes.segments[0]
    y_plots = []
    spike_count = populationx.get_spike_counts()
    print("spike_count X: ", spike_count)
    base_rate = get_base_rate(populationx)
    print("base_rate: ", base_rate)

    if title != "train":
        date = save_date()
        f = open(storage_url +"logs/"+ snn_name + "" + date + "-landmark_output_spikesx.txt", "w+")
        for ind, spiketrain in enumerate(x_spikes.spiketrains):
            f.write("[" + str(ind) + ",")
            for spike in spiketrain:
                x_plots.append([ind, spike])
                f.write(" " + str(spike) + ", ")
            f.write("], \n")
        f.close()
        f.close()
        f = open(storage_url + "logs/"+ snn_name + ""  + date + "-landmark_output_spikesy.txt", "w+")

        for ind, spiketrain in enumerate(y_spikes.spiketrains):
            f.write("[" + str(ind) + ",")
            for spike in spiketrain:
                y_plots.append([ind, spike])
                f.write(" " + str(spike) + ", ")
            f.write("], \n")
        f.close()
        spike_count = populationy.get_spike_counts()
        print("spike_count Y: ", spike_count)
        base_rate = get_base_rate(populationy)
        print("base_rate: ", base_rate)
        for x in x_plots:
            plt.scatter(x[1], x[0])
        plt.xlabel('Spike Time')
        plt.ylabel('Neuron Index')
        plt.title('location output X')
        plt.savefig(storage_url+"plots/"+ snn_name + "" + date + "x_location_positions.jpg")
        plt.show()
        for y in y_plots:
            plt.scatter(y[1], y[0])
        plt.xlabel('Spike Time')
        plt.ylabel('Neuron Index')
        plt.title('location output Y')
        plt.savefig(storage_url+"plots/"+ snn_name + "" + date + "y_location_positions.jpg")
        plt.show()

        try:
            f = open(storage_url+"logs/"+ snn_name + "" + date + "-loc_x_output_spikes.txt", "w+")
            counter = 0
            for x in x_plots:
                f.write("["+str(x[0])+", ")
                for x in x_plots:
                    if x[0] == counter:
                        f.write("" + str(x[1]) + ",")
                f.write("],  \n")
                counter+=1
            f.close()

            f = open(storage_url+"logs/"+ snn_name + "" + date + "-loc_y_output_spikes.txt", "w+")
            counter = 0
            for y in y_plots:
                f.write("[" + str(y[0]) + ", ")
                for y in y_plots:
                    if y[0] == counter:
                        f.write("" + str(y[1]) + ",")
                f.write("],  \n")
                counter += 1
            f.close()
        except:
            print("save Failed")

#plot SNN mapping results
def plot_landmarks(populationx, title, snn_name = ""):
    spike_count = populationx.get_spike_counts()
    print("spike_count: ", spike_count)
    get_base_rate(populationx)
    l_spikes = populationx.get_data('spikes')
    l_spikes = l_spikes.segments[0]
    l_plots = []

    if title != "train":
        date = save_date()

        f = open(storage_url+"logs/"+ snn_name + "" + date + "-landmark_output_spikes.txt", "w+")
        for ind, l_spiketrain in enumerate(l_spikes.spiketrains):
            f.write("[" + str(ind) + ", ")
            for spike in l_spiketrain:
                l_plots.append([ind, spike])
                f.write(", " + str(spike) + "")
            f.write("], \n")
        f.close()

        for l in l_plots:
            plt.scatter(l[1], l[0])
        plt.xlabel('Spike Time')
        plt.ylabel('Neuron Index')
        plt.title('Mapping Landmarks')
        plt.savefig(storage_url+"plots/"+ snn_name + ""+date+"landmarks_plots.jpg")
        plt.show()
#plot SNN mapping positions: currently not quite fully functioning
def plot_landmarks_pos(populationx, populationy, title, snn_name = ""):
    spike_count = populationx.get_spike_counts()
    print("spike_count: ", spike_count)
    x_base = get_base_rate(populationx)
    x_spikes = populationx.get_data('spikes')
    x_spikes = x_spikes.segments[0]
    x_plots = []

    spike_count = populationy.get_spike_counts()
    print("spike_count: ", spike_count)
    y_base = get_base_rate(populationy)
    y_spikes = populationy.get_data('spikes')
    y_spikes = y_spikes.segments[0]
    y_plots = []

    if title != "train":
        date = save_date()

        for ind, spiketrain in enumerate(x_spikes.spiketrains):
            counter = 0
            for spike in spiketrain:
                counter+=1
                x_plots.append([ind, spike])
        for ind, spiketrain in enumerate(y_spikes.spiketrains):
            counter = 0
            for spike in spiketrain:
                counter+=1
                y_plots.append([ind, spike])
        for y in y_plots:
            plt.scatter(y[0], y[1])
        plt.xlabel('Spike Time')
        plt.ylabel('Neuron Index')
        plt.title('Mapping Landmarks')
        plt.savefig(storage_url+"plots/"+ snn_name + ""+date+"landmarks_positionsy.jpg")
        plt.show()

        for x in x_plots:
            plt.scatter(x[0], x[1])
        plt.xlabel('Spike Time')
        plt.ylabel('Neuron Index')
        plt.title('Mapping Landmarks')
        plt.savefig(storage_url+"plots/"+ snn_name + ""+date+"landmarks_positionsx.jpg")
        plt.show()
#get SNN average spikes
def get_base_rate(population):
    mean_spike = population.mean_spike_count(gather=True)
    print("mean_spike: ", mean_spike)
    return mean_spike

def log_land_det(log, title):
    datet = save_date()

    f = open("log_data/training_data/" + title + "-" +datet+".py", "w+")
    for line in log:
        f.write("["+str(line)+"], \n")
    f.close()
# save SNN results at each time step
def save_snn_results(data, name):
    date = save_date()
    f = open(""+storage_url+"logs/"+name+"/" + date + "-"+name+"-SNN_RESULTS.txt", "w+")
    f.write("# SNN Location output")
    for line in data:
        f.write("[ time: " + str(line[0]) + ", x Neuron: " + str(line[1][0]) + ", y Neuron: " + str(line[1][1]) + "], \n")
    f.write("# SNN Location end")
    f.close()
# Plot localisation output upon the track example
def plot_activated_loc_neuron(track, results, name): #populationx, populationy,
    plt.ylim(600, 0)
    plt.xlim(0, 1200)

    x = []
    y = []
    for item in track:
        x.append(item.position[0][0])
        y.append(item.position[1][1])
    plt.scatter(x, y, s=3, c='grey')

    for result in results:
        x_pos = result[1][0] * 5
        y_pos = result[1][1] * 5
        rectangle = plt.Rectangle((x_pos, y_pos), 5, 5, fc='blue', ec="red")
        plt.gca().add_patch(rectangle)
    date = save_date()

    plt.title("Localisation: "+name+"")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.savefig(storage_url+"plots/"+name+"/localisation/" + date + "_loc_neurons.jpg")
    plt.legend(["Track Objects"])
    plt.show()
# plot landmark SNN outputs
def plot_seen_landmarks(pop, title, total_time, time_step, current_time, name):
    all_activity = []
    colours = [ 'green', 'blue', 'orange', 'red', 'purple', 'pink', 'grey']
    while current_time < total_time:
        result = get_active_neurons(pop, current_time, time_step)
        all_activity.append([result, current_time])
        c = 0
        result.sort(key=lambda result: result[2])
        for x in result:

            plt.scatter(current_time, x[0], color=colours[c])
            if c != 6:
                c+=1
        current_time += time_step
    date = save_date()
    plt.ylabel("Neuron Index")
    plt.xlabel("Spike Time")
    plt.title("Landmarks Seen: "+name+"")
    handles = []
    for i in colours:
        handles.append(plt.scatter(0, 0, 10, i))
    plt.legend(handles, ["landmark 1", "landmark 2", "landmark 3", "landmark 4", "landmark 5", "landmark 6", "Probable outliers"])
    plt.savefig(storage_url+"plots/"+name+"/mapping/" + date + "_"+title+"_active_neurons_and_landmarks_"+name+".jpg")
    plt.show()
    f = open(storage_url+"logs/"+name+"/mapping/" + date + "-_active_neurons_and_landmarks"+name+".txt", "w+")
    f.write("# SNN landmarks output")
    for line in all_activity:
        for l in line[0]:
            f.write(
                "[ time: " + str(line[1]) + ", ind " + str(l[0]) + ", count: " + str(l[1]) + "], \n")
    f.write("# SNN landmarks end")
    f.close()
# plot activated neurons
def plot_active_neurons(pop, title, total_time, time_step, current_time, name):
    all_activity = []
    colours = [ 'green', 'blue', 'orange', 'red', 'purple', 'pink', 'grey']
    while current_time < total_time:
        result = get_active_neurons(pop, current_time, time_step)
        all_activity.append([result,current_time])
        result.sort(key=lambda result: result[2])
        c = 0
        for x in result:
            plt.scatter(x[2], x[0], color=colours[c])
            if c != 6:
                c+=1
        current_time += time_step
    date = save_date()
    plt.title(title.replace("_", " ")+": "+name)
    plt.ylabel("Neuron Index")
    plt.xlabel("Spike Time")
    handles = []
    for i in colours:
        handles.append(plt.scatter(0,0,10,i))
    plt.legend(handles, ["Spike 1", "Spike 2", "Spike 3", "Spike 4", "spike 5", "spike 6", "Probable outliers"])
    plt.savefig(storage_url+"plots/"+name+"/"+date+"_"+title+"_active_neurons"+name+".jpg")
    plt.show()
    f = open(storage_url+"logs/"+name+"/"+date+"_"+title+"_active_neurons_"+name+".txt", "w+")
    f.write("# SNN "+title+" output")
    for line in all_activity:
        for l in line[0]:
            f.write(
                "[ time: " + str(line[1]) + ", ind " + str(l[0]) + ", count: " + str(l[1]) + "], \n")
    f.write("# SNN "+title+" end")
    f.close()
#gets active neurons between time steps
def get_active_neurons(pop, current_time, time_step):
    spikes = pop.get_data('spikes')
    spikes = spikes.segments[0]
    spike_list = []
    for ind, spiketrain in enumerate(spikes.spiketrains):
        counter = 0
        for spike in spiketrain:
            if (current_time - time_step) < spike < current_time:
                counter += 1
        if counter > 0:
            spike_list.append([ind, counter, spike])
    return spike_list
# plot grid example of the 5 by 5 pixel X & Y representation
def plot_environment(track,name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ylim(600, 0)
    plt.xlim(0, 1200)

    xminor_ticks = np.arange(0, 1200, 5)
    yminor_ticks = np.arange(0, 600, 5)
    x = []
    y = []
    for item in track:
        x.append(item.position[0][0])
        y.append(item.position[1][1])
    ax.scatter(x, y, s=3, c='grey')
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(yminor_ticks, minor=True)
    ax.grid(which='both')

    plt.title("Localisation: "+name+"")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.savefig(storage_url+"plots/grid_example_square.jpg")
    plt.legend(["Track Objects"])
    plt.show()