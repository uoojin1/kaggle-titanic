## Titanic: Machine Learning from Disaster

### Competition Description

The
sinking of the RMS Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the Titanic sank after colliding
with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational
tragedy shocked the international community and led to better safety regulations
for ships.

One of the reasons that the shipwreck led to such loss of life was
that there were not enough lifeboats for the passengers and crew. Although there
was some element of luck involved in surviving the sinking, some groups of
people were more likely to survive than others, such as women, children, and the
upper-class.

In this challenge, we ask you to complete the analysis of what
sorts of people were likely to survive. In particular, we ask you to apply the
tools of machine learning to predict which passengers survived the tragedy.

<img src="images/titanic-sinking.jpg" height="auto" width="auto"></img>

## 2. Collecting the data

training and testing data set can be obtained from
https://www.kaggle.com/c/titanic/data

```{.python .input  n=7}
# Load training and testing dataset with pandas
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

```{.python .input  n=8}
# Let's peek on the training data
train.head()
```

```{.json .output n=8}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>PassengerId</th>\n      <th>Survived</th>\n      <th>Pclass</th>\n      <th>Name</th>\n      <th>Sex</th>\n      <th>Age</th>\n      <th>SibSp</th>\n      <th>Parch</th>\n      <th>Ticket</th>\n      <th>Fare</th>\n      <th>Cabin</th>\n      <th>Embarked</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>0</td>\n      <td>3</td>\n      <td>Braund, Mr. Owen Harris</td>\n      <td>male</td>\n      <td>22.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>A/5 21171</td>\n      <td>7.2500</td>\n      <td>NaN</td>\n      <td>S</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>1</td>\n      <td>1</td>\n      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n      <td>female</td>\n      <td>38.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>PC 17599</td>\n      <td>71.2833</td>\n      <td>C85</td>\n      <td>C</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>1</td>\n      <td>3</td>\n      <td>Heikkinen, Miss. Laina</td>\n      <td>female</td>\n      <td>26.0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>STON/O2. 3101282</td>\n      <td>7.9250</td>\n      <td>NaN</td>\n      <td>S</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>1</td>\n      <td>1</td>\n      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n      <td>female</td>\n      <td>35.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>113803</td>\n      <td>53.1000</td>\n      <td>C123</td>\n      <td>S</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>0</td>\n      <td>3</td>\n      <td>Allen, Mr. William Henry</td>\n      <td>male</td>\n      <td>35.0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>373450</td>\n      <td>8.0500</td>\n      <td>NaN</td>\n      <td>S</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "   PassengerId  Survived  Pclass  \\\n0            1         0       3   \n1            2         1       1   \n2            3         1       3   \n3            4         1       1   \n4            5         0       3   \n\n                                                Name     Sex   Age  SibSp  \\\n0                            Braund, Mr. Owen Harris    male  22.0      1   \n1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n2                             Heikkinen, Miss. Laina  female  26.0      0   \n3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n4                           Allen, Mr. William Henry    male  35.0      0   \n\n   Parch            Ticket     Fare Cabin Embarked  \n0      0         A/5 21171   7.2500   NaN        S  \n1      0          PC 17599  71.2833   C85        C  \n2      0  STON/O2. 3101282   7.9250   NaN        S  \n3      0            113803  53.1000  C123        S  \n4      0            373450   8.0500   NaN        S  "
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}

```