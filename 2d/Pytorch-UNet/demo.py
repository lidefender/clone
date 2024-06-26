from kaggle_secrets import UserSecretsClient
secret_label = "your-secret-label"
secret_value = UserSecretsClient().get_secret(secret_label)