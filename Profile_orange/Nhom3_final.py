import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import statsmodels.api as sm

# Đọc dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv('Orange Quality Data.csv')
    # Đổi tên cột sang tiếng Việt
    df.columns = ['Kích thước (cm)', 'Trọng lượng (g)', 'Độ ngọt (Brix)', 'Độ chua (pH)', 
                  'Độ mềm (1-5)', 'Thời gian thu hoạch (ngày)', 'Độ chín (1-5)', 
                  'Màu sắc', 'Giống', 'Khuyết tật (C/K)', 'Chất lượng (1-5)']
    return df

# Tạo các tính năng mới
def create_features(df):
    df['Tỉ lệ đường/axit'] = df['Độ ngọt (Brix)'] / df['Độ chua (pH)']
    df['Chỉ số trưởng thành'] = df['Độ chín (1-5)'] * df['Thời gian thu hoạch (ngày)']
    df['Mật độ'] = df['Trọng lượng (g)'] / (df['Kích thước (cm)'] ** 3)
    return df

# Chuẩn bị dữ liệu
def prepare_data(df):
    df = create_features(df)
    numeric_df = df.select_dtypes(include=[np.number])
    X = numeric_df.drop('Chất lượng (1-5)', axis=1)
    y = numeric_df['Chất lượng (1-5)']
    return X, y

# Hàm phân loại cam
def classify_orange(size, weight, brix, ph):
    if brix < 6 or ph > 4.4 or size < 6 or weight < 100:
        return "Cam hư"
    elif size >= 8 and weight >= 200:
        if brix >= 11 and ph <= 3.5:
            return "Cam to ngọt"
        else:
            return "Cam to chua"
    elif size < 8 or weight < 200:
        if brix >= 11 and ph <= 3.5:
            return "Cam nhỏ ngọt"
        else:
            return "Cam nhỏ chua"
    else:
        return "Cam cân bằng"

# Nhập dữ liệu từ người dùng
def user_input_features():
    size = st.sidebar.slider('Kích thước (cm)', 6.0, 10.0, 8.0, key="size_slider")
    weight = st.sidebar.slider('Trọng lượng (g)', 100, 300, 200, key="weight_slider")
    brix = st.sidebar.slider('Độ ngọt (Brix)', 6.0, 16.0, 11.0, key="brix_slider")
    ph = st.sidebar.slider('Độ chua (pH)', 2.8, 4.4, 3.5, key="ph_slider")
    softness = st.sidebar.slider('Độ mềm (1-5)', 1, 5, 3, key="softness_slider")
    harvest_time = st.sidebar.slider('Thời gian thu hoạch (ngày)', 4, 25, 14, key="harvest_time_slider")
    ripeness = st.sidebar.slider('Độ chín (1-5)', 1, 5, 3, key="ripeness_slider")
    
    data = {'Kích thước (cm)': size,
            'Trọng lượng (g)': weight,
            'Độ ngọt (Brix)': brix,
            'Độ chua (pH)': ph,
            'Độ mềm (1-5)': softness,
            'Thời gian thu hoạch (ngày)': harvest_time,
            'Độ chín (1-5)': ripeness}
    features = pd.DataFrame(data, index=[0])
    
    # Tạo các tính năng mới
    features = create_features(features)
    
    return features

# Hàm tìm hyperparameters tối ưu và huấn luyện mô hình
def optimize_model(model, param_grid, X_train, y_train):
    try:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    except Exception as e:
        st.warning(f"Lỗi khi tối ưu hóa mô hình: {str(e)}")
        return model

# Hàm đánh giá mô hình
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    return {'MSE': mse, 'MAE': mae, 'R2': r2, 'CV MSE': cv_mse}

# Huấn luyện mô hình
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grids = {
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'SVR': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    }
    
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'KNN': KNeighborsRegressor(),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Neural Network': MLPRegressor(max_iter=5000, random_state=42, early_stopping=True, validation_fraction=0.1)
    }
    
    results = {}
    for name, model in models.items():
        if name != 'Linear Regression':
            model = optimize_model(model, param_grids[name], X_train_scaled, y_train)
        results[name] = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        models[name] = model
    
    return scaler, models, results, X_train_scaled, X_test_scaled, y_train, y_test

# Định nghĩa các emoji cho mỗi mô hình
model_icons = {
    'Random Forest': '🌳',
    'KNN': '🔍',
    'Linear Regression': '📊',
    'SVR': '🎯',
    'Gradient Boosting': '🚀',
    'Neural Network': '🧠'
}

def tune_best_model(X_train, y_train, X_test, y_test, best_model_name, models):
    best_model = models[best_model_name]
    
    if best_model_name == 'Random Forest':
        param_distributions = {
            'n_estimators': randint(100, 500),
            'max_depth': [None] + list(randint(10, 100).rvs(10)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    elif best_model_name == 'Gradient Boosting':
        param_distributions = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    elif best_model_name == 'Neural Network':
        param_distributions = {
            'hidden_layer_sizes': [(randint(50, 200).rvs(), randint(50, 200).rvs()) for _ in range(10)],
            'alpha': uniform(0.0001, 0.01),
            'learning_rate_init': uniform(0.001, 0.1)
        }
    else:
        st.warning(f"Không có cấu hình tinh chỉnh cho mô hình {best_model_name}")
        return best_model, None

    random_search = RandomizedSearchCV(best_model, param_distributions, n_iter=50, cv=5, 
                                       scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    
    tuned_model = random_search.best_estimator_
    
    # Evaluate tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    mse_tuned = mean_squared_error(y_test, y_pred_tuned)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    
    return tuned_model, {'MSE': mse_tuned, 'R2': r2_tuned}

def display_tuning_results(best_model_name, original_results, tuned_results=None):
    # Thêm dòng chỉ ra mô hình tốt nhất
    st.write(f"**Mô hình tốt nhất là: {best_model_name}**")

    st.subheader(f'Kết quả cho mô hình {best_model_name}')
    
    # Tạo DataFrame để so sánh với các giá trị số
    comparison_df = pd.DataFrame({
        'Mô hình gốc': [original_results['MSE']/25*100, original_results['R2']*100]
    }, index=['MSE (%)', 'R2 (%)'])
    
    if tuned_results:
        comparison_df['Mô hình tinh chỉnh'] = [tuned_results['MSE']/25*100, tuned_results['R2']*100]
        
        # Tính tỷ lệ phần trăm cải thiện
        mse_improvement = (original_results['MSE'] - tuned_results['MSE']) / original_results['MSE'] * 100
        r2_improvement = (tuned_results['R2'] - original_results['R2']) / original_results['R2'] * 100
        
        st.write(f"Cải thiện MSE: {mse_improvement:.2f}%")
        st.write(f"Cải thiện R2: {r2_improvement:.2f}%")
    else:
        st.write("Mô hình này không được tinh chỉnh.")
    
    # Hiển thị bảng so sánh với tỷ lệ phần trăm được định dạng
    st.table(comparison_df.style.format("{:.2f}%"))
    
    # Tạo biểu đồ thanh để so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    comparison_df.plot(kind='bar', ax=ax1, ylabel='MSE (%)')
    ax1.set_title('So sánh MSE')
    ax1.legend(title='')
    ax1.set_ylim(0, 100)  # Đặt giới hạn trục y cho MSE
    
    comparison_df.plot(kind='bar', ax=ax2, ylabel='R2 (%)')
    ax2.set_title('So sánh R2')
    ax2.legend(title='')
    ax2.set_ylim(0, 100)  # Đặt giới hạn trục y cho R2
        
# Thêm nhãn phần trăm trên đầu mỗi thanh
    for ax in [ax1, ax2]:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%')
    
    plt.tight_layout()
    st.pyplot(fig)
    
#Phân tích chi tiết cho mô hình 
def detailed_model_analysis(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    residuals = y_test - y_pred
    
    st.subheader(f'Phân tích chi tiết cho mô hình {model_name}')
    
    # Metrics
    st.write(f"Mean Squared Error (MSE): {mse/25*100:.2f}%")
    st.write(f"Mean Absolute Error (MAE): {mae/5*100:.2f}%")
    st.write(f"R-squared (R2) Score: {r2*100:.2f}%")
    
    # Residual Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals)
    ax.set_xlabel('Giá trị Dự đoán')
    ax.set_ylabel('Residuals')
    ax.set_title('Biểu đồ Residual')
    ax.axhline(y=0, color='r', linestyle='--')
    st.pyplot(fig)
    
    # Q-Q Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sm.qqplot(residuals, ax=ax, line='45')
    ax.set_title('Q-Q Plot của Residuals')
    st.pyplot(fig)
    
    # Histogram của Residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=30)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Tần suất')
    ax.set_title('Histogram của Residuals')
    st.pyplot(fig)
    
    # Feature Importance (nếu có)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)
    
    # Ghi chú giải thích
    st.info("""
    Giải thích các chỉ số:
    
    1. Mean Squared Error (MSE): Thể hiện trung bình bình phương của sai số. Giá trị càng thấp càng tốt. 
       MSE được tính là phần trăm của giá trị tối đa có thể (25 cho thang điểm 1-5).
    
    2. Mean Absolute Error (MAE): Thể hiện trung bình của giá trị tuyệt đối của sai số. Giá trị càng thấp càng tốt.
       MAE được tính là phần trăm của thang điểm tối đa (5).
    
    3. R-squared (R2) Score: Thể hiện phần trăm biến thiên của biến phụ thuộc được giải thích bởi mô hình. 
       Giá trị càng gần 100% càng tốt.
    
    Các biểu đồ:
    - Biểu đồ Residual: Điểm phân tán đều quanh đường y=0 cho thấy mô hình có độ chính xác tốt.
    - Q-Q Plot: Nếu các điểm nằm gần đường chéo, residuals có phân phối gần với phân phối chuẩn.
    - Histogram của Residuals: Hình dạng chuông đối xứng cho thấy residuals có phân phối gần với phân phối chuẩn.
    """)
def main():
    st.title('Dự đoán và Phân loại Chất lượng Cam')

    # Đọc và chuẩn bị dữ liệu
    df = load_data()
    X, y = prepare_data(df)

    # Huấn luyện mô hình
    scaler, models, results, X_train_scaled, X_test_scaled, y_train, y_test = train_models(X, y)

    # Chọn mô hình với icon
    st.sidebar.header('Chọn Mô hình Dự đoán')
    model_options = [f"{model_icons[model]} {model}" for model in models.keys()]
    selected_model_with_icon = st.sidebar.selectbox('Chọn mô hình', model_options, label_visibility="collapsed")
    selected_model = selected_model_with_icon.split(' ', 1)[1]  # Lấy tên mô hình từ chuỗi đã chọn

    # Giao diện người dùng để nhập thông số
    st.sidebar.header('Nhập thông số Cam')
    input_df = user_input_features()

    st.subheader('Thông số đầu vào')
    st.write(input_df)

    # Kết quả dự đoán và phân loại
    input_scaled = scaler.transform(input_df)
    prediction = models[selected_model].predict(input_scaled)[0]
    
    size = input_df['Kích thước (cm)'].values[0]
    weight = input_df['Trọng lượng (g)'].values[0]
    brix = input_df['Độ ngọt (Brix)'].values[0]
    ph = input_df['Độ chua (pH)'].values[0]
    classification = classify_orange(size, weight, brix, ph)
    
    st.subheader('Kết quả dự đoán')
    st.write(f"Mô hình được chọn: {model_icons[selected_model]} {selected_model}")
    st.write(f"Chất lượng cam được dự đoán là: {prediction:.2f}/5.00 ({prediction/5.0*100:.2f}%)")
    st.write(f"Dựa trên các thông số, cam này được phân loại là: {classification}")

    # Đếm số lượng cam trong mỗi loại
    df['Phân loại'] = df.apply(lambda row: classify_orange(
        row['Kích thước (cm)'], 
        row['Trọng lượng (g)'], 
        row['Độ ngọt (Brix)'], 
        row['Độ chua (pH)']), axis=1)
    classification_counts = df['Phân loại'].value_counts()
    
    st.subheader('Số lượng Cam theo Phân loại')
    st.write(classification_counts)

    # Biểu đồ phân phối phân loại
    st.subheader('Biểu đồ Phân loại')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=classification_counts.index, y=classification_counts.values, ax=ax)
    plt.title('Biểu đồ Phân loại Cam')
    plt.xlabel('Phân loại')
    plt.ylabel('Số lượng')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Kết quả training (dự đoán chi tiết)
    st.subheader('Kết quả Training')

    # Dự đoán chi tiết
    for name, model in models.items():
        prediction = model.predict(input_scaled)
        st.write(f'{name}: {prediction[0]:.2f}/5.0 ({prediction[0]/5.0*100:.2f}%)')

    # Tầm quan trọng của các đặc trưng
    st.subheader('Tầm quan trọng của đặc tính')
    feature_importance = pd.DataFrame({
        'Đặc tính': X.columns,
        'Random Forest': models['Random Forest'].feature_importances_,
        'Gradient Boosting': models['Gradient Boosting'].feature_importances_
    })

    # Thêm tầm quan trọng của Linear Regression
    lr_importance = np.abs(models['Linear Regression'].coef_)
    feature_importance['Linear Regression'] = lr_importance / np.sum(lr_importance)

    feature_importance = feature_importance.sort_values('Random Forest', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    feature_importance.plot(x='Đặc tính', y=['Random Forest', 'Gradient Boosting', 'Linear Regression'], kind='bar', ax=ax)
    plt.title('Tầm quan trọng của đặc tính')
    plt.xlabel('Đặc tính')
    plt.ylabel('Tầm quan trọng')
    plt.legend(title='Mô hình')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Thêm giải thích về tầm quan trọng của đặc tính
    st.info("""
    Giải thích về Tầm quan trọng của đặc tính:

    1. Ý nghĩa: Biểu đồ này thể hiện mức độ ảnh hưởng của từng đặc tính đến dự đoán chất lượng cam trong các mô hình khác nhau.

    2. Cách đọc biểu đồ:
    - Trục x: Liệt kê các đặc tính của cam.
    - Trục y: Thể hiện mức độ quan trọng của mỗi đặc tính (từ 0 đến 1).
    - Các cột màu: Mỗi màu đại diện cho một mô hình khác nhau.

    3. Diễn giải:
    - Đặc tính có cột cao hơn có ảnh hưởng lớn hơn đến dự đoán chất lượng cam.
    - Các mô hình khác nhau có thể đánh giá tầm quan trọng của đặc tính khác nhau.
    
       Ứng dụng:
    - Giúp xác định các yếu tố quan trọng nhất ảnh hưởng đến chất lượng cam.
    - Có thể sử dụng để tối ưu hóa quy trình trồng và chăm sóc cam.
    - Hỗ trợ việc lựa chọn đặc tính khi thu thập dữ liệu trong tương lai.

       Lưu ý: 
    - Tầm quan trọng có thể thay đổi giữa các mô hình.
    - Nên xem xét kết hợp kết quả từ nhiều mô hình để có cái nhìn toàn diện.
""")
     #Hiển thị top 3 đặc tính quan trọng nhất
    top_features = feature_importance.nlargest(3, 'Random Forest')
    st.write("Top 3 đặc tính quan trọng nhất (dựa trên Random Forest):")
    for index, row in top_features.iterrows():
        st.write(f"- {row['Đặc tính']}: {row['Random Forest']:.4f}")

    # Biểu đồ tương quan chi tiết
    st.subheader('Biểu đồ Tương quan Chi tiết')
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, 
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar=False)
    plt.title('Biểu đồ Tương quan Chi tiết', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    # Thêm giải thích về biểu đồ tương quan
    st.info("""
    Giải thích về Biểu đồ Tương quan Chi tiết:

    1. Ý nghĩa:
    - Biểu đồ này thể hiện mức độ tương quan giữa các đặc tính của cam.
    - Tương quan dao động từ -1 đến 1, trong đó:
        * 1 là tương quan dương hoàn hảo
        * -1 là tương quan âm hoàn hảo
        * 0 là không có tương quan

    2. Cách đọc biểu đồ:
    - Mỗi ô trong biểu đồ thể hiện tương quan giữa hai đặc tính.
    - Màu sắc: 
        * Màu đỏ thể hiện tương quan dương
        * Màu xanh thể hiện tương quan âm
        * Màu càng đậm, tương quan càng mạnh
    - Số trong mỗi ô là giá trị cụ thể của hệ số tương quan.

    3. Diễn giải:
    - Tương quan gần 1 hoặc -1: Hai đặc tính có mối liên hệ mạnh.
    - Tương quan gần 0: Hai đặc tính ít hoặc không có liên hệ.
    - Dấu (+/-) chỉ ra chiều của mối quan hệ (cùng chiều hoặc ngược chiều).
            
    4. Ứng dụng:
    - Hiểu rõ mối quan hệ giữa các đặc tính của cam.
    - Phát hiện các đặc tính có thể dự đoán lẫn nhau.
    - Hỗ trợ trong việc lựa chọn đặc tính cho mô hình dự đoán.

    5. Lưu ý: 
    - Tương quan không đồng nghĩa với quan hệ nhân quả.
    - Cần xem xét ý nghĩa thực tế của mối tương quan, không chỉ dựa vào số liệu.
            """)
    # Hiển thị các cặp đặc tính có tương quan mạnh nhất
    strong_correlations = corr.unstack().sort_values(key=abs, ascending=False)
    strong_correlations = strong_correlations[(strong_correlations != 1.0) & (abs(strong_correlations) > 0.5)]
    st.write("Các cặp đặc tính có tương quan mạnh (|r| > 0.5):")
    for (feature1, feature2), correlation in strong_correlations.items():
        if feature1 != feature2:
            st.write(f"- {feature1} và {feature2}: {correlation:.2f}")

    # Bảng so sánh các mô hình
    st.subheader('So sánh các Mô hình')
    comparison_df = pd.DataFrame(results).T

    # Chuyển đổi MSE và MAE thành phần trăm so với giá trị trung bình của y
    y_mean = y.mean()
    comparison_df['MSE (%)'] = (comparison_df['MSE'] / (y_mean ** 2)) * 100
    comparison_df['MAE (%)'] = (comparison_df['MAE'] / y_mean) * 100

    # Chuyển đổi R2 thành phần trăm
    comparison_df['R2 (%)'] = comparison_df['R2'] * 100

    # Chuyển đổi CV MSE thành phần trăm
    comparison_df['CV MSE (%)'] = (comparison_df['CV MSE'] / (y_mean ** 2)) * 100

    # Sắp xếp theo MSE (%) tăng dần
    comparison_df = comparison_df.sort_values('MSE (%)')

    # Chọn các cột cần hiển thị
    display_columns = ['MSE (%)', 'MAE (%)', 'R2 (%)', 'CV MSE (%)']

    # Tạo bảng so sánh với màu sắc
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(comparison_df[display_columns], annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
    plt.title('So sánh các Mô hình (%)', fontsize=16)
    plt.ylabel('Mô hình')
    plt.tight_layout()
    st.pyplot(fig)

    # Thêm ghi chú
    st.info("Ghi chú: MSE và MAE thấp hơn là tốt hơn, trong khi R2 cao hơn là tốt hơn.")

    # Hiển thị mô hình tốt nhất
    best_model = comparison_df.index[0]
    st.write(f"Mô hình tốt nhất: {best_model}")
    st.write(f"MSE: {comparison_df.loc[best_model, 'MSE (%)']: .2f}%")
    st.write(f"R2: {comparison_df.loc[best_model, 'R2 (%)']: .2f}%")

    # Biểu đồ so sánh MSE và R2 của các mô hình
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    comparison_df['MSE (%)'].plot(kind='bar', ax=ax1)
    ax1.set_title('MSE của các mô hình (%)')
    ax1.set_ylabel('MSE (%)')
    ax1.set_xlabel('Mô hình')

    comparison_df['R2 (%)'].plot(kind='bar', ax=ax2)
    ax2.set_title('R2 của các mô hình (%)')
    ax2.set_ylabel('R2 (%)')
    ax2.set_xlabel('Mô hình')

    plt.tight_layout()
    st.pyplot(fig)

    # Thêm ghi chú cho biểu đồ
    st.info("Ghi chú: Đối với biểu đồ MSE, cột thấp hơn là tốt hơn. Đối với biểu đồ R2, cột cao hơn là tốt hơn.")

    # Biểu đồ Dự đoán vs Thực tế cho mỗi mô hình
    st.subheader('Biểu đồ Dự đoán vs Thực tế')
    fig, axs = plt.subplots(3, 2, figsize=(20, 24))  # Tăng kích thước để chứa 6 mô hình
    for (name, model), ax in zip(models.items(), axs.ravel()):
        y_pred = model.predict(X_test_scaled)
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Giá trị Thực tế')
        ax.set_ylabel('Giá trị Dự đoán')
        ax.set_title(f'Biểu đồ Dự đoán vs Thực tế - {name}')
        ax.text(0.05, 0.95, f'R2: {r2_score(y_test, y_pred):.2f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
    plt.tight_layout()
    st.pyplot(fig)

    # Thêm ghi chú giải thích
    st.info("""
    Ghi chú về Biểu đồ Dự đoán vs Thực tế:
    - Mỗi điểm trên biểu đồ đại diện cho một mẫu cam trong tập kiểm tra.
    - Trục x thể hiện giá trị chất lượng thực tế của cam.
    - Trục y thể hiện giá trị chất lượng dự đoán bởi mô hình.
    - Đường chéo màu đỏ đứt nét thể hiện dự đoán hoàn hảo (giá trị dự đoán = giá trị thực tế).
    - Các điểm càng gần đường chéo đỏ, dự đoán càng chính xác.
    - R2 (hệ số xác định) càng gần 1, mô hình càng tốt.
    - Mô hình lý tưởng sẽ có các điểm tập trung sát đường chéo đỏ và R2 gần 1.
    """)
    # Phần tinh chỉnh mô hình
    st.subheader('Tinh chỉnh Mô hình')

    if st.button('Hiển thị kết quả mô hình tốt nhất'):
        with st.spinner('Đang xử lý...'):
            best_model_name = comparison_df.index[0]
            original_results = {
                'MSE': comparison_df.loc[best_model_name, 'MSE'],
                'R2': comparison_df.loc[best_model_name, 'R2']
            }
            
            if best_model_name != 'Linear Regression':
                tuned_model, tuned_results = tune_best_model(X_train_scaled, y_train, X_test_scaled, y_test, best_model_name, models)
                
                if tuned_results:
                    display_tuning_results(best_model_name, original_results, tuned_results)
                    # Sử dụng mô hình đã tinh chỉnh cho phân tích chi tiết
                    detailed_model_analysis(tuned_model, X_test_scaled, y_test, f"{best_model_name} (Tinh chỉnh)")
                else:
                    st.warning(f"Không thể tinh chỉnh mô hình {best_model_name}")
                    display_tuning_results(best_model_name, original_results)
                    # Sử dụng mô hình gốc cho phân tích chi tiết
                    detailed_model_analysis(models[best_model_name], X_test_scaled, y_test, best_model_name)
            else:
                st.info("Mô hình Linear Regression không cần tinh chỉnh.")
                display_tuning_results(best_model_name, original_results)
                # Phân tích chi tiết cho Linear Regression
                detailed_model_analysis(models[best_model_name], X_test_scaled, y_test, best_model_name)
    

def display_designer_info():
    st.markdown("---")
    st.markdown("### Người thiết kế: Nhóm 3")
    st.markdown("Phạm Phúc Khang-2275201140014")
    st.markdown("Hoàng Qúy Cường-2275201140007")
    st.markdown("Hoàng Gia Anh-2275201140001")

if __name__ == '__main__':
    main()
    display_designer_info()