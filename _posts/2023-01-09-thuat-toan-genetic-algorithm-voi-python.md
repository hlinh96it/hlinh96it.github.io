---
title: Giải thuật di truyền (Genetic Algorithm) với Python (Phần 1)
author: hoanglinh
categories: [Optimization Algorithms]
tags: [meta-heuristics algorithm]
math: true
img_path: posts_media/2023-01-09-thuat-toan-genetic-algorithm-voi-python/
image:
  path: ga-cover.jpeg
---

Từ lâu, chúng ta thường được nghe rằng loài người được tiến hóa từ loài vượn cổ. Quan điểm này xuất phát từ thuyết tiến hóa của Charles Darwin. Song song với thuyết tiến hóa của Darwin - quan điểm chọn lọc tự nhiên thì còn 1 thuyết tiến hóa nữa của Jean-Baptiste Lamarck. Tuy nhiên trong bài viết này mình sẽ không đề cập tới lĩnh vực này mà mình muốn giới thiệu tới bạn đọc một thuật toán sử dụng chọn lọc tự nhiên để tìm ra lời giải tối ưu đối với các bài toán có không gian tìm kiếm rất lớn - **Genetic Algorithm (GA) - *Giải thuật Di Truyền*.**

# Ý tưởng từ sự tiến hóa

Lấy ví dụ ngay là con người như chúng ta, giả thiết rằng vì đặc tính săn bắt và hái lượm, các bộ phận trên cơ thể tổ tiên chúng ta đã có sự thay đổi hay tiến hóa để phù hợp với sự phát triển hơn. Điều này thể hiện rõ ngay cả đối với thế hệ ba mẹ và con cái, thường con cái sẽ nhận được các gen tốt từ cả bố và mẹ và từ đó thế hệ sau sẽ càng tốt hơn so với thế hệ trước.

Nói theo một cách khác, quá trình tiến hóa gồm 4 thành phần chính: quần thể, đột biến, sinh sản và chọn lọc tự nhiên. Lấy cảm hứng từ tự nhiên, thuật toán GA cũng bao gồm 4 thành phần này. Các thành phần đều có mối liên hệ mật thiết với nhau và cùng với mục đích chung là tạo ra các thế hệ con có các đặc tính tốt hơn thế hệ trước, trong đó:

1.  **Population - quần thể**: Một quần thể ban đầu sẽ có những cá thể nhất định với những đặc tính khác nhau, những đặc tính này sẽ quy định khả năng sinh sản, sinh tồn, khả năng đáp ứng điều kiện môi trường của từng cá thể.
2.  **Selection - chọn lọc tự nhiên**: Theo thời gian những cá thể yếu hơn, không có khả năng sinh tồn sẽ bị loại bỏ bởi những tác nhân như tranh chấp chuỗi thức ăn, môi trường tác độc, bị loài khác tiêu diệt, … Cuối cùng sẽ còn lại những cá thể có đặc tính ưu việt hơn sẽ được giữ lại $\rightarrow$ **Adaptive individual**. 
3.  **Mutation - đột biến**: Như chúng ta đã biết thì mỗi cá thể con được sinh ra sẽ được kế thừa lại những đặc tính của cả cha và mẹ. Tuy nhiên, để đáp ứng được với sự thay đổi của môi trường hoặc thậm chí là tiến hóa, đột biến là quá trình cần thiết và đóng vai trò then chốt cho quá trình **Selection**
4.  **Evolution - tiến hóa**: Những cá thể đột biến không phải luôn là những cá thể mạnh mẽ và có đủ khả năng sinh tồn, **Chọn lọc tự nhiên** sẽ chọn ra những cá thể đột biến nhưng có thể thích nghi với môi trường sống tốt hơn những cá thể khác trong quần thể. Sau một thời gian sinh sản, những gen đột biến sẽ chiếm ưu thế và chiếm đa số trong quần thể.

# Lý thuyết thuật toán GA

Không giống như hầu hết các thuật toán tối ưu hóa, thuật toán di truyền không sử dụng đạo hàm để tìm cực tiểu (local minima). Một trong những ưu điểm quan trọng nhất của thuật toán GA là khả năng tìm ra điểm global minima mà không bị mắc kẹt trong local minima. Tính ngẫu nhiên (randomness) đóng một vai trò quan trọng trong cấu trúc của thuật toán di truyền và đó là lý do chính khiến thuật toán di truyền tiếp tục tìm kiếm không gian tìm kiếm.

Mặc dù thuật toán GA đã xuất hiện từ lâu, hiện tại có rất nhiều phiên bản cải tiến thuật toán để phù hợp với các bài toán khác nhau, tuy nhiên, nhìn chung đều gồm các bước cơ bản như hình dưới: 

![genetic-algorithm-flowchart](ga-flowchart.png)_Flowchart of genetic algorithms [2]_

Thuật toán GA tạo ra một quần thể (population) ban đầu gồm các giải candidate solutions được tạo ngẫu nhiên, các candidate solutions này được đánh giá và fitness value của chúng được tính toán bằng một hàm nào đó. Giá trị fitness value của một solution là giá trị xác định mức độ tốt của solution đó. Tùy vào bài toán tìm max hoặc min, fitness value càng cao/thấp thì solution càng tốt.

Lấy ví dụ với bài toán tính tổng sao cho lớn nhất, mỗi solution gồm 4 tham số, fitness value được tính bằng cách cộng tất cả các tham số lại, thể hiện như hình dưới:

![sum-problem](sum-problem.png)_An example of a generation_

Nếu thuật toán chưa gặp phải điều kiện dừng như đã chạy đủ số lần lặp, hay fitness value không tốt hơn nữa, thuật toán di truyền sẽ tạo ra thế hệ tiếp theo. Hoạt động di truyền đầu tiên là **Chọn lọc**; trong hoạt động này, các cá thể chuyển sang thế hệ tiếp theo được chọn. Sau quá trình lựa chọn, hoạt động ghép nối bắt đầu. Hoạt động ghép đôi sẽ ghép đôi từng cá thể được chọn cho hoạt động **Mating operation**. **Mating operation** lấy các cá thể bố mẹ đã ghép đôi và tạo ra **thế hệ con (offsprings)**, chúng sẽ **thay thế** các cá thể không được chọn trong thao tác Chọn lọc, vì vậy thế hệ tiếp theo có cùng số lượng cá thể như thế hệ trước. Quá trình này được lặp lại cho đến khi các tiêu chí chấm dứt được đáp ứng.

# Code GA from scratch with Python

Trong bài viết này, chúng ta sẽ cùng code GA từ đầu bằng `Python` và thư viện `Numpy`. Mỗi hoạt động di truyền được đề cập ở trên sẽ được viết lại dưới dạng các function. Trước khi bắt đầu với mã thuật toán di truyền, chúng ta cần nhập một số thư viện dưới dạng:

```python
import numpy as np
import random	
from random import gauss, randrange
```

## Khởi tạo individual và population

Thuật toán GA bắt đầu quá trình tối ưu hóa bằng cách tạo ra một tập hợp ban đầu các candidate individuals có mã gen được tạo ngẫu nhiên

```python
def create_individual(num_gens, upper_limit, lower_limit):
    return [round(random.random() * (upper_limit - lower_limit) + lower_limit, 1) \
        for x in range(num_gens)]
```

Sau đó, ta cần tạo một quần thể (population) bao gồm các individual:

```python
def population(number_of_individuals, number_of_genes, upper_limit, lower_limit):
    return [individual(number_of_genes, upper_limit, lower_limit) \
        for x in range(number_of_individuals)]
```

## Tính toán fitness value

Sau khi gọi 2 function trên, population ban đầu sẽ được tạo ra. Thuật toán GA tạo ra thế hệ đầu tiên, các fitness value của các individual được tính toán.

>  Hàm `fitness_calculation` xác định fitness value của mỗi `individual`, cách tính fitness value tùy thuộc vào từng bài toán. Nếu vấn đề là tối ưu hóa các tham số của một hàm, thì hàm đó nên được triển khai thành hàm tính toán fitness.
{: .prompt-tip }

Để đơn giản, chúng ta sẽ xem xét ví dụ tạo được đưa ra ở đầu bài viết, tính tổng của mỗi individual.

```python
def fitness_calculation(individual):
    return sum(individual)
```

> Trong trường hợp này, bài toán chỉ đơn giản là có duy nhất một parameter. Tuy nhiên, đối với các bài toán có nhiều parameters, ví dụ như revenue của một công ty được tính bởi hàm của nhiều factors khác nhau. Nếu các factors không đồng nhất về range, ví dụ tiền (triệu đồng) với số lượng đơn hàng (chục cái), thuật toán sẽ khó tìm được solution tốt. Ta cần chú ý đến **normalize** các tham số trước khi tính fitness value.
{: .prompt-info}

Bên cạnh đó, ngoài cách tính fitness như trên, ta có thể đặt mức độ quan trọng có mỗi gene khác nhau, gọi là weight $w_i$ và $g_i$ là gene ở vị trí $i$. Trong đó, tổng weight phải bằng 1:

$$
\text{Fitness value}= w_1g_1 + w_2g_2 + ... + w_ng_n \tag{1}
$$

> Trong một số trường hợp khác yêu cầu tối ưu tất cả parameters cùng nhau, những bài toán đó thuộc về **multi-objectives optimization problem**, bạn đọc có thể tìm hiểu thêm tại bài viết này [Bài toán multi-objective optimization](https://developer.twitt)
{: .prompt-tip }

## Quá trình Selection

Quá trình selection lấy đầu vào là tất cả các individual và các giá trị fitness của chúng. Output là tập hợp các individuals được chọn cho quá trình update ở generation tiếp theo. Câu hỏi là làm sao để chọn? Thuật ngữ **“elitism”** hay gọi là **“tinh hoa”** được sử dụng như là phương pháp để chọn các individual có fitness value tốt nhất, từ đó ta đảm bảo được việc bỏ sót các solution tốt.

Phương pháp **Roulette wheel selection** là một trong số các phương pháp “elitism” để chọn individual. Mỗi individual đều có xác xuất được chọn để update tùy thuộc vào giá trị fitness value của chúng, fitness value cao đồng nghĩa với xác xuất được chọn cao hơn. Ví dụ như hình dưới:

![ga-selection](ga-selection.png)_Roulette wheel selection figure_

Phương pháp Roulette wheel selection lấy tổng tích lũy (cumulative sums) và tính toán xác xuất được chọn của mỗi individual, từ đó return các individual được chọn. Bằng cách tính tổng tích lũy, mỗi individual có một giá trị duy nhất từ 0 đến 1. Để chọn các individual, một số từ 0 đến 1 được tạo ngẫu nhiên và individual gần với số được tạo ngẫu nhiên sẽ được chọn. Hàm roulette có thể được viết là

```python
def roulette_selection(cum_sum, chance):
    variable = list(cum_sum.copy())
    variable.append(chance)
    variable = sorted(variable)
    
    return variable.index(chance)
```

Phương pháp **Fittest half selection**: lấy một nửa các individual tốt nhất trong cả population

![ga-fittest](ga-fittest-half.png)_Fittest half selection_

Phương pháp **Random Selection :**  individuals được chọn randomly.

![ga-random](ga-random-selection.png)_Random selection_

```python
def individual_selection(generation, method='fittest-half'):
    selected_individuals = {}
    selected_range = int(len(generation['individuals']) // 2)

    generation['normalized_fitness'] = sorted(
        [generation['fitness'][x] / sum(generation['fitness']) \
         for x in range(len(generation['fitness']))], reverse=True
    )

    # calculate cumulative sum of normalized fitness array
    generation['cum_sum'] = np.array(generation['normalized_fitness']).cumsum()

    if method == 'roulette-wheel':
        # select half of population
        selected_individuals = []

        for x in range(selected_range):
            selected_individuals.append(roulette(generation['cum_sum'], random.random()))

            # check if there are some duplicated individuals
            while len(set(selected_individuals)) != len(selected_individuals):
                selected_individuals[x] = roulette(generation['cum_sum'], random.random())

        selected_individuals = {
            'individuals': [generation['individuals'][selected_individuals[idx]] for idx in range(selected_range)],
            'fitness': [generation['fitness'][selected_individuals[idx]] for idx in range(selected_range)]
        }

    elif method == 'fittest-half':
        selected_individuals = {
            'individuals': [generation['individuals'][idx] for idx in range(selected_range)],
            'fitness': [generation['fitness'][idx] for idx in range(selected_range)]
        }

    elif method == 'random':
        random_inds = random.sample(range(len(generation['individuals'])), selected_range)
        selected_individuals = {
            'individuals': [generation['individuals'][idx] for idx in random_inds],
            'fitness': [generation['fitness'][idx] for idx in random_inds]
        }

    return selected_individuals
```

## Quá trình pairing and mating

Tương tự như quá trình **selection**, quá trình **pairing** và **mating** cũng có các phương pháp chọn khác nhau. Trước tiên, chúng ta sẽ thảo luận về 3 phương pháp **pairing**:

-  **Fittest:** trong phương pháp này, các cá nhân được ghép đôi từng cặp một, bắt đầu từ individual có fitness value tốt nhất. Bằng cách đó, những cá thể khỏe mạnh hơn được ghép cặp với nhau, nhưng những cá thể kém khỏe mạnh hơn cũng được ghép cặp với nhau.

-  **Ngẫu nhiên**: Trong phương pháp này, các cá nhân được ghép đôi một cách ngẫu nhiên.

-  **Weighted random**: Trong phương pháp này, các cá nhân được ghép cặp ngẫu nhiên từng đôi một, nhưng những cá thể phù hợp hơn có cơ hội được chọn để ghép cặp cao hơn.

   ![pairing-fittest](ga-pairing-weighted.webp)_Pairing follow Weighted random strategy_

Quá trình pairing có thể viết như sau:

```python
def pairing(elite, selected_inds, method='weighted-random'):
    individuals = [elite['individuals']] + selected_inds['individuals']
    fitness = [elite['fitness']] + selected_inds['fitness']
    parent = []

    pairing_len = len(individuals) // 2
    if method == 'random':

        for x in range(pairing_len):
            parent.append([
                individuals[random.randint(0, len(individuals) - 1)],
                individuals[random.randint(0, len(individuals) - 1)]
            ])

            while parent[x][0] == parent[x][1]:
                parent[x][1] = individuals[random.randint(0, len(individuals) - 1)]

    elif method == 'weighted-random':
        normalized_fitness = sorted(
            [fitness[x] / sum(fitness) for x in range(pairing_len)], reverse=True
        )
        cum_sum = np.array(normalized_fitness).cumsum()

        for x in range(pairing_len):
            parent.append(
                [individuals[roulette(cum_sum, random.random())],
                 individuals[roulette(cum_sum, random.random())]]
            )
            while parent[x][0] == parent[x][1]:
                parent[x][1] = individuals[roulette(cum_sum, random.random())]

    return parent
```

Sau khi đã chọn được cặp, bây giờ ta phải quyết định kết hợp thế nào. Đối với GA, có 2 cách để kết hợp phổ biến là single và 2 points. Đối với phương pháp single point, 2 gens hoán đổi kết hợp phần trước của bố và phần sau điểm cắt của gene mẹ. Phương pháp 2 points cũng tương tự như vậy.

![ga-mating](ga-mating-multi-points.png){: width="500"}_Phương pháp multiple points mating_

Quá trình mating có thể viết lại như sau:

```python
def mating(parents, method='single-point'):
    offsprings = []
    if method == 'single-point':
        pivot_point = random.randint(1, len(parents[0]))
        offsprings = [parents[0]
                      [0:pivot_point] + parents[1][pivot_point:], parents[1]
                      [0:pivot_point] + parents[0][pivot_point:]]

    if method == 'multiple-points':
        pivot_point_1 = random.randint(1, len(parents[0] - 1))
        pivot_point_2 = random.randint(1, len(parents[0]))

        while pivot_point_2 < pivot_point_1:
            pivot_point_2 = random.randint(1, len(parents[0]))
        offsprings = \
            [parents[0][0:pivot_point_1] + parents[1][pivot_point_1:pivot_point_2] +
             [parents[0][pivot_point_2:]], [parents[1][0:pivot_point_1] +
                                            parents[0][pivot_point_1:pivot_point_2] +
                                            [parents[1][pivot_point_2:]]]]

    return offsprings
```

## Quá trình mutation tạo ra đột biến

Đột biến ngẫu nhiên xảy ra ở các cá thể được chọn và con cái của chúng để cải thiện sự đa dạng của thế hệ tiếp theo. Nếu có tinh hoa (elite) trong thuật toán di truyền, cá thể tinh hoa không trải qua đột biến ngẫu nhiên nên chúng ta không mất solution tốt nhất. Chúng ta sẽ thảo luận về hai phương pháp đột biến khác nhau.

-  **Gauss**: Trong phương pháp này, gen trải qua đột biến được thay thế bằng một số được tạo ra theo phân bố gauss xung quanh gen ban đầu.
-  **Reset**: Trong phương pháp này, gen ban đầu được thay thế bằng gen được tạo ngẫu nhiên

![ga-mutation](ga-mutation.webp){: width="600"}_Quá trình mutation followed by reset strategy_

Quá trình mutation có thể được viết lại như sau:

```python
def mutation(individual, upper_limit, lower_limit, muatation_rate=2,
             method='Reset', standard_deviation=0.001):
    gene = [np.random.randint(0, 7)]

    for x in range(muatation_rate - 1):
        gene.append(np.random.randint(0, 7))
        while len(set(gene)) < len(gene):
            gene[x] = np.random.randint(0, 7)
    mutated_individual = individual.copy()

    if method == 'Gauss':
        for x in range(muatation_rate):
            mutated_individual[x] = round(individual[x] + gauss(0, standard_deviation), 1)
    if method == 'Reset':
        for x in range(muatation_rate):
            mutated_individual[x] = round(np.random.random() * (upper_limit - lower_limit) + lower_limit, 1)

    return mutated_individual
```

## Tạo ra các thế hệ tiếp theo

Thế hệ tiếp theo được tạo ra bằng cách sử dụng các hoạt động di truyền mà chúng ta đã thảo luận. Elitsm có thể được đưa vào thuật toán di truyền trong quá trình tạo ra thế hệ tiếp theo. 

```python
def next_generation(gen, upper_limit, lower_limit):
    elit = {}
    next_gen = {}
    elit['individuals'] = gen['individuals'].pop(-1)
    elit['fitness'] = gen['fitness'].pop(-1)

    selected = individual_selection(gen)
    parents = pairing(elit, selected)
    offsprings = [[[mating(parents[x]) for x in range(len(parents))][y][z] for z in range(2)] \
                  for y in range(len(parents))]

    offsprings1 = [offsprings[x][0] for x in range(len(parents))]
    offsprings2 = [offsprings[x][1] for x in range(len(parents))]

    unmutated = selected['individuals'] + offsprings1 + offsprings2
    mutated = [mutation(unmutated[x], upper_limit, lower_limit) for x in range(len(gen['individuals']))]

    unsorted_individuals = mutated + [elit['individuals']]
    unsorted_next_gen = [fitness_calculation(mutated[x]) for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x] for x in range(len(gen['fitness']))] + [elit['fitness']]
    sorted_next_gen = sorted([[unsorted_individuals[x], unsorted_fitness[x]] \
                              for x in range(len(unsorted_individuals))], key=lambda x: x[1])

    next_gen['individuals'] = [sorted_next_gen[x][0] for x in range(len(sorted_next_gen))]
    next_gen['fitness'] = [sorted_next_gen[x][1] for x in range(len(sorted_next_gen))]

    gen['individuals'].append(elit['individuals'])
    gen['fitness'].append(elit['fitness'])

    return next_gen
```

## Termination Criteria

Sau khi một thế hệ được tạo, các tiêu chí kết thúc (termination criteria) được sử dụng để xác định xem thuật toán GA có nên tạo một thế hệ khác hay không. Các tiêu chí kết thúc khác nhau có thể được sử dụng đồng thời và nếu thuật toán di truyền thỏa mãn một trong các tiêu chí thì thuật toán di truyền dừng lại. Chúng ta sẽ thảo luận về 4 tiêu chí:

- **Đạt giá trị fitness tối đa** : Tiêu chí chấm dứt này kiểm tra xem best individual trong thế hệ hiện tại có đáp ứng các tiêu chí của chúng ta hay không. Sử dụng phương pháp này có thể thu được kết quả mong muốn. Như được thấy từ hình bên dưới, các giá trị có thể được xác định bao gồm một số cực tiểu cục bộ hoặc global minima.

  ![local-vs-global](local-vs-absolute-extrema.png)

- **Giá trị fitness trung bình tối đa**: Nếu chúng ta quan tâm đến một tập hợp các solution, các giá trị trung bình của các individuals trong các thế hệ hiện tại có thể được kiểm tra để xác định xem thế hệ hiện tại có đáp ứng mong đợi của chúng ta hay không.

- **Số thế hệ tối đa (number of generations)**: Chúng ta có thể giới hạn số thế hệ tối đa (hay số lần chạy) được tạo bởi giải thuật di truyền.

- **Số lượng thể lực tương tự tối đa**: Do cá thể tốt nhất trong một thế hệ ưu tú chuyển sang thế hệ tiếp theo mà không bị đột biến. Cá nhân này cũng có thể là cá nhân tốt nhất trong thế hệ tiếp theo. Chúng ta có thể giới hạn số lượng để cùng một cá thể trở thành cá thể tốt nhất vì điều này có thể nói rằng thuật toán di truyền bị mắc kẹt trong một cực tiểu cục bộ. Hàm để kiểm tra xem giá trị phù hợp tối đa có thay đổi hay không có thể được viết là

```python
def fitness_similarity_chech(max_fitness, number_of_similarity):
    result = False
    similarity = 0
    for n in range(len(max_fitness)-1):
        if max_fitness[n] == max_fitness[n+1]:
            similarity += 1
        else:
            similarity = 0
    if similarity == number_of_similarity-1:
        result = True
    return result
```

## Running the Algorithm

Bây giờ tất cả function chúng ta cần cho thuật toán di truyền đã sẵn sàng, chúng ta có thể bắt đầu quá trình tối ưu hóa. Để chạy thuật toán di truyền với 20 cá thể trong mỗi thế hệ:

  ```python
  result_file = 'ga_result.txt'
  
  def first_generation(pop):
      fitness = [fitness_calculation(pop[x]) for x in range(len(pop))]
      sorted_fitness = sorted([[pop[x], fitness[x]] for x in range(len(pop))], key=lambda x: x[1])
      population = [sorted_fitness[x][0] for x in range(len(sorted_fitness))]
      fitness = [sorted_fitness[x][1] for x in range(len(sorted_fitness))]
  
      return {'individuals': population, 'fitness': fitness}
  
  
  pop = population(number_of_individuals=20, number_of_genes=10, upper_limit=1, lower_limit=0)
  gen = [first_generation(pop)]
  fitness_avg = np.array([sum(gen[0]['fitness']) / len(gen[0]['fitness'])])
  fitness_max = np.array([max(gen[0]['fitness'])])
  
  res = open(result_file, 'a')
  res.write('\n' + str(gen) + '\n')
  res.close()
  
  finish = False
  while not finish:
      if max(fitness_max) > 8 or max(fitness_avg) > 7 or fitness_similarity_check(fitness_max, 50):
          break
  
      gen.append(next_generation(gen[-1], 1, 0))
      fitness_avg = np.append(fitness_avg, sum(gen[-1]['fitness']) / len(gen[-1]['fitness']))
      fitness_max = np.append(fitness_max, max(gen[-1]['fitness']))
      res = open(result_file, 'a')
      res.write('\n' + str(gen[-1]) + '\n')
      res.close()
  ```

# Conclusion

Thuật toán di truyền GA có thể được sử dụng để giải quyết các vấn đề tối ưu hóa ràng buộc đa tham số. Giống như hầu hết các thuật toán tối ưu hóa, thuật toán di truyền có thể được triển khai trực tiếp từ một số thư viện như `sklearn`, nhưng việc code lại thuật toán từ đầu sẽ giúp ta hiểu về cách thức hoạt động của thuật toán và cần phải  được điều chỉnh cho phù hợp với một vấn đề cụ thể hoặc cái bài toán tối ưu khác nhau.

Cảm ơn bạn đã đọc, tôi hy vọng bài viết hữu ích. Nếu bạn có câu thắc mắc nào hãy để lại ở dưới phần bình luận bên dưới nhé!

> Bạn đọc có thể tìm thấy code full của mình tại repo [đây](https://gist.github.com/hlinh96it/2a96fa4e7bc4d6f83a52633f8a77c956).
{: .prompt-info}

# References

1.  <https://nerophung.github.io/2020/05/28/genetic-algorithm>
2.  <https://towardsdatascience.com/continuous-genetic-algorithm-from-scratch-with-python-ff29deedd099>
3.  <https://mitpress.mit.edu/books/adaptation-natural-and-artificial-systems>