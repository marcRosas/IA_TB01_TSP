import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

#clase para las ciudades
class Ciudad:
    def __init__(self, nombre, x, y):
        self.nombre = nombre
        self.x = x
        self.y = y

    def distancia(self, ciudad):
        DistanciaX = abs(self.x - ciudad.x)
        DistanciaY = abs(self.y - ciudad.y)
        distancia = np.sqrt((DistanciaX ** 2) + (DistanciaY ** 2))
        return distancia

    def __repr__(self):
        return "(" + str(self.nombre) + ")"


# Crea la función fitness
class Fitness:
    def __init__(self, ruta):
        self.ruta = ruta
        self.distancia = 0
        self.fitness = 0.0

    def ruta_Distancia(self):
        if self.distancia == 0:
            camino = 0
            for i in range(0, len(self.ruta)):
                DesdeCiudad = self.ruta[i]
                HastaCiudad = None
                if i + 1 < len(self.ruta):
                    HastaCiudad = self.ruta[i + 1]
                else:
                    HastaCiudad = self.ruta[0]
                camino += DesdeCiudad.distancia(HastaCiudad)
            self.distancia = camino
        return self.distancia

    def rutaFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.ruta_Distancia())
        return self.fitness


#Crea la ruta
#Este metodo crea las ciudades en modo aleatorio
def CrearRuta(ListaCiudades):
    ruta = random.sample(ListaCiudades, len(ListaCiudades))
    return ruta


#Crea la poblacion inicial
#Crea la población de modo aleatorio con un tamaño especificado
def PoblacionInicial(tamanio, ListaCiudades):
    poblacion = []

    for i in range(0, tamanio):
        poblacion.append(CrearRuta(ListaCiudades))
    return poblacion


# Create the genetic algorithm
# Rank individuals
# This function takes a poblacion and orders it in descending order using the fitness of each individual
def RutasTop(poblacion):
    fitnessResults = {}
    for i in range(0, len(poblacion)):
        fitnessResults[i] = Fitness(poblacion[i]).rutaFitness()
    resultados_comb = sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)
    return resultados_comb


# Función de selección que se usara para la lista de rutas principales
def seleccion(popRank, TamanioElite):
    ResultadosSeleccion = []
    df = pd.DataFrame(np.array(popRank), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, TamanioElite):
        ResultadosSeleccion.append(popRank[i][0])
    for i in range(0, len(popRank) - TamanioElite):
        pick = 100 * random.random()
        for i in range(0, len(popRank)):
            if pick <= df.iat[i, 3]:
                ResultadosSeleccion.append(popRank[i][0])
                break
    return ResultadosSeleccion


#Crea un conjunto de apareamiento
def matingPool(poblacion, ResSelec):
    matingpool = []
    for i in range(0, len(ResSelec)):
        index = ResSelec[i]
        matingpool.append(poblacion[index])
    return matingpool


# Funcion cruzada de 2 padres para generar hijos
def breed(padre1, padre2):
    Hijo = []
    HijoP1 = []
    HijoP2 = []

    geneA = int(random.random() * len(padre1))
    geneB = int(random.random() * len(padre2))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        HijoP1.append(padre1[i])

    HijoP2 = [item for item in padre2 if item not in HijoP1]
    print(startGene, endGene)

    print(padre1)
    print(padre2)

    print(HijoP1)
    print(HijoP2)
    Hijo = HijoP1 + HijoP2

    print(Hijo)
    return Hijo


#Función para ejecutar el cruce sobre el grupo de acoplamiento completo

def breedPopulation(matingpool, eliteSize):
    hijo = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        hijo.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        hijo.append(child)
    return hijo


#Crea una función para mutar una ruta
def mutate(individual, RatioMutac):
    for swapped in range(len(individual)):
        if (random.random() < RatioMutac):
            swapWith = int(random.random() * len(individual))

            ciudad1 = individual[swapped]
            ciudad2 = individual[swapWith]

            individual[swapped] = ciudad2
            individual[swapWith] = ciudad1
    return individual


# Funcion para correr la mutacion de la poblacion
def mutatePopulation(poblacion, RatioMutac):
    mutatedPop = []

    for ind in range(0, len(poblacion)):
        mutatedInd = mutate(poblacion[ind], RatioMutac)
        mutatedPop.append(mutatedInd)
    return mutatedPop


#Se unen todos los pasos para la prox generación
def ProxGen(currentGen, eliteSize, RatioMutac):
    popRanked = RutasTop(currentGen)
    selectionResults = seleccion(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    hijo = breedPopulation(matingpool, eliteSize)
    ProxGen = mutatePopulation(hijo, RatioMutac)
    return ProxGen


#Finalmente se genera el algoritmo genético
def geneticAlgorithm(poblacion, tamanio, eliteSize, RatioMutac, generations):
    pop = PoblacionInicial(tamanio, poblacion)
    progress = [1 / RutasTop(pop)[0][1]]
    print("distancia inicial: " + str(progress[0]))

    for i in range(1, generations + 1):

        pop = ProxGen(pop, eliteSize, RatioMutac)
        progress.append(1 / RutasTop(pop)[0][1])
        if i % 50 == 0:
            print('Generación ' + str(i), "Distancia: ", progress[i])

    Indice_MejorRuta = RutasTop(pop)[0][0]
    MejorRuta = pop[Indice_MejorRuta]

    plt.plot(progress)
    plt.ylabel('Distancia')
    plt.xlabel('Generación')
    plt.title('Fitness vs Generación')
    plt.tight_layout()
    plt.show()

    return MejorRuta


#Implementación del alg genetico
#Crea la lista de ciudades
ListaCiudades = []

for i in range(0,5):
    ListaCiudades.append(Ciudad(nombre = i, x=int(random.random() * 200), y=int(random.random() * 200)))


mejor_ruta=geneticAlgorithm(poblacion=ListaCiudades, tamanio=30, eliteSize=20, RatioMutac=0.01, generations=1)
x=[]
y=[]
for i in mejor_ruta:
  x.append(i.x)
  y.append(i.y)
x.append(mejor_ruta[0].x)
y.append(mejor_ruta[0].y)
plt.plot(x, y, '--o')
plt.xlabel('X')
plt.ylabel('Y')
ax=plt.gca()
plt.title('Ruta Final')
bbox_props = dict(boxstyle="circle,pad=0.3", fc='C0', ec="black", lw=0.5)
for i in range(1,len(ListaCiudades)+1):
  ax.text(ListaCiudades[i-1].x, ListaCiudades[i-1].y, str(i), ha="center", va="center",
            size=8,
            bbox=bbox_props)
plt.tight_layout()
plt.show()
