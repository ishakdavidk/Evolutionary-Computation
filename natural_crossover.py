import numpy as np
import random
import operator


def breed(parent1, parent2, cityList, groups):

    # Generating child
    available_num_city = len(cityList)

    child = []
    group = 0
    while available_num_city > 0:
        i = 0
        j = 0
        assign = True
        cont = False
        assign_previous = False
        parent_select = group % 2
        while assign or cont or assign_previous:
            if available_num_city == 0:
                break

            if len(groups[group]) == 0:
                break

            if i == len(cityList) or j == -len(cityList):
                break

            if assign == False and cont == False and assign_previous == False:
                break

            # Even groups for parent 1
            if parent_select == 0:
                idx = np.where(groups[group] == np.array(parent1[i]))

                if len(idx[0]) == 1:
                    if assign == True or cont == True:
                        child.append(parent1[i])
                        del groups[group][idx[0][0]]

                        available_num_city -= 1

                        assign = False
                        cont = True

                    if i == 0:
                        assign_previous = True
                        assign_idx = len(child) - 1
                else:
                    cont = False

                if len(groups[group]) == 0:
                    break

                if assign_previous == True:
                    j -= 1
                    idx_prev = np.where(groups[group] == np.array(parent1[j]))

                    if len(idx_prev[0]) == 1:
                        child.insert(assign_idx, parent1[j])
                        del groups[group][idx_prev[0][0]]

                        available_num_city -= 1
                    else:
                        assign_previous = False

            # Odd groups for parent 2
            if parent_select == 1:
                idx = np.where(groups[group] == np.array(parent2[i]))

                if len(idx[0]) == 1:
                    if assign == True or cont == True:
                        child.append(parent2[i])
                        del groups[group][idx[0][0]]

                        available_num_city -= 1

                        assign = False
                        cont = True

                    if i == 0:
                        assign_previous = True
                        assign_idx = len(child) - 1
                else:
                    cont = False

                if len(groups[group]) == 0:
                    break

                if assign_previous == True:
                    j -= 1
                    idx_prev = np.where(groups[group] == np.array(parent2[j]))

                    if len(idx_prev[0]) == 1:
                        child.insert(assign_idx, parent2[j])
                        del groups[group][idx_prev[0][0]]

                        available_num_city -= 1
                    else:
                        assign_previous = False

            i += 1

        if group < len(groups) - 1:
            group += 1
        else:
            group = 0

    return child


def natural_breed(parent1, parent2, cityList, num_groups_in):
    # generate number of the group
    if num_groups_in == None:
        max_num_groups = round(len(cityList) / 2)
        num_groups = random.randint(1, max_num_groups)
    else:
        num_groups = num_groups_in

    # print('num_groups = ', num_groups)

    # assign number of the members for each group
    groups = []
    total_num_members = 0
    for i in range(num_groups, 0, -1):
        available_num_city = len(cityList) - total_num_members

        if i > 1:
            max_num_members = available_num_city - (i - 1)  # every group should have, at least, one member
            num_members = random.randint(1, max_num_members)
            total_num_members += num_members
            groups.append([num_members])
        else:
            groups.append([available_num_city])

    # assigning cities to each group
    city_list_assign = cityList.copy()
    for i in range(len(groups)):
        for j in range(groups[i][0]):
            city_idx = random.randint(0, len(city_list_assign) - 1)

            groups[i].append(city_list_assign[city_idx])
            del city_list_assign[city_idx]

        # print('group_', i, ' : ', groups[i])
        del groups[i][0]

    # Generating child
    child = breed(parent1, parent2, cityList, groups)

    return child


def routeDistance(route):
    pathDistance = 0
    for i in range(0, len(route)):
        fromCity = route[i]
        toCity = None
        if i + 1 < len(route):
            toCity = route[i + 1]
        else:
            toCity = route[0]
        pathDistance += fromCity.distance(toCity)
    return  pathDistance


def semi_natural_breed(parent1, parent2, cityList, num_groups_in): # This is selective crossover
    # generate number of the group
    if num_groups_in == None:
        max_num_groups = round(len(cityList) / 8)
        num_groups = random.randint(2, max_num_groups+1)
    else:
        num_groups = num_groups_in

    # print('num_groups = ', num_groups)

    # assign number of the members for each group
    groups = []
    num_members = round((3 * len(cityList)) / 4)    # group 0 has 75% of the total cities
    total_num_members = num_members
    groups.append([num_members])

    for i in range(num_groups-1, 0, -1):
        available_num_city = len(cityList) - total_num_members

        if i > 1:
            max_num_members = available_num_city - (i - 1)  # every group should have, at least, one member
            num_members = random.randint(1, max_num_members)
            total_num_members += num_members
            groups.append([num_members])
        else:
            groups.append([available_num_city])

    # assigning cities to each group
    parent1_dist = routeDistance(parent1)
    parent2_dist = routeDistance(parent2)

    if parent1_dist >= parent2_dist:
        parent1, parent2 = parent2, parent1

    city_list_assign = parent1.copy()

    city_idx = random.randint(0, len(city_list_assign) - 1)
    for i in range(groups[0][0]):
        if city_idx >= len(city_list_assign):
            city_idx = 0

        groups[0].append(city_list_assign[city_idx])
        del city_list_assign[city_idx]

    del groups[0][0]

    for i in range(len(groups)-1):
        for j in range(groups[i+1][0]):
            city_idx = random.randint(0, len(city_list_assign) - 1)

            groups[i+1].append(city_list_assign[city_idx])
            del city_list_assign[city_idx]

        # print('group_', i, ' : ', groups[i])
        del groups[i+1][0]

    # Generating child
    child = breed(parent1, parent2, cityList, groups)

    return child


def breed2(parent1, parent2, cityList, groups):

    # Generating child
    available_num_city = len(cityList) - len(groups[0])

    child = groups[0][:]
    del groups[0][:]

    group = 1
    i = 0
    j = 0
    assign_previous = False
    while available_num_city > 0:
        if available_num_city == 0:
            break

        if i == len(cityList) or j == -len(cityList):
            break

        # Odd groups for parent 2
        idx = np.where(groups[group] == np.array(parent2[i]))

        if len(idx[0]) == 1:
            child.append(parent2[i])
            del groups[group][idx[0][0]]

            available_num_city -= 1

            if i == 0:
                assign_previous = True
                assign_idx = len(child) - 1

        if len(groups[group]) == 0:
            break

        if assign_previous == True:
            j -= 1
            idx_prev = np.where(groups[group] == np.array(parent2[j]))

            if len(idx_prev[0]) == 1:
                child.insert(assign_idx, parent2[j])
                del groups[group][idx_prev[0][0]]

                available_num_city -= 1
            else:
                assign_previous = False

        i += 1

    return child


def semi_natural_breed2(parent1, parent2, cityList): # This is selective crossover
    # assign number of the members for each group
    groups = []
    num_members0 = round((1 * len(cityList)) / 5)    # group 0 has 20% of the total cities
    groups.append([num_members0])

    num_members1 = len(cityList) - num_members0
    groups.append([num_members1])

    city_idx_max = len(cityList) - groups[0][0]
    index_to = groups[0][0]

    dist = float('inf')
    city_idx_final = None
    parent_select = 0
    for _ in range(3):
        city_idx1 = random.randint(0, city_idx_max)
        candidate1 = parent1[city_idx1:city_idx1 + index_to]
        candidate_dist1 = routeDistance(candidate1)

        city_idx2 = random.randint(0, city_idx_max)
        candidate2 = parent2[city_idx2:city_idx2 + index_to]
        candidate_dist2 = routeDistance(candidate2)

        if candidate_dist1 < dist:
            groups[0] = candidate1
            city_idx_final = city_idx1
            dist = candidate_dist1
            parent_select = 1

        if candidate_dist2 < dist:
            groups[0] = candidate2
            city_idx_final = city_idx2
            dist = candidate_dist2
            parent_select = 2

    if parent_select == 2:
        parent1, parent2 = parent2, parent1

    city_list_assign = parent1.copy()

    del city_list_assign[city_idx_final:city_idx_final + index_to]

    groups[1] = city_list_assign[:]
    del city_list_assign[:]

    # Generating child
    child = breed2(parent1, parent2, cityList, groups)

    return child