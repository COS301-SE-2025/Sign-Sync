import React from "react";
import { Link } from "react-router-dom";

class LoginPage extends React.Component 
{
    handleSubmit = (e) => 
    {
        e.preventDefault();
        console.log("Form submitted");
    };

    render() 
    {
        return (
            <div className="flex relative justify-center items-center w-screen h-screen bg-blue-900">
                <div className="absolute bg-sky-950 h-[906px] rounded-[55px] w-[1237px] z-[1] max-md:h-4/5 max-md:w-[90%] max-sm:h-3/4 max-sm:w-[95%]" />

                <form
                    onSubmit={this.handleSubmit}
                    className="box-border flex relative flex-col gap-12 items-start p-12 bg-white rounded-xl border border-amber-50 border-solid h-[796px] min-w-80 w-[773px] z-[2] max-md:gap-8 max-md:p-8 max-md:h-auto max-md:max-w-[600px] max-md:w-[90%] max-sm:gap-6 max-sm:p-6 max-sm:max-w-[400px] max-sm:w-[95%]"
                >
                    <div className="flex flex-col gap-4 items-start w-full">
                        <label className="self-stretch text-3xl font-bold leading-10 text-stone-900 max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Username
                        </label>
                        <input
                            type="text"
                            placeholder="Enter username"
                            className="self-stretch px-8 py-6 text-2xl font-bold leading-4 bg-white rounded-lg border border-solid flex-[1_0_0] min-w-60 text-zinc-900 max-md:px-6 max-md:py-5 max-md:text-sm max-sm:px-5 max-sm:py-4 max-sm:text-sm"
                            name="username"
                            required
                        />
                    </div>

                    <div className="flex flex-col gap-4 items-start w-full">
                        <label className="self-stretch text-3xl font-bold leading-10 text-stone-900 max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Email
                        </label>
                        <input
                            type="email"
                            placeholder="Enter email"
                            className="self-stretch px-8 py-6 text-2xl font-bold leading-4 bg-white rounded-lg border border-solid flex-[1_0_0] min-w-60 text-zinc-900 max-md:px-6 max-md:py-5 max-md:text-sm max-sm:px-5 max-sm:py-4 max-sm:text-sm"
                            name="email"
                            required
                        />
                    </div>

                    <div className="flex flex-col gap-4 items-start w-full">
                        <label className="self-stretch text-3xl font-bold leading-10 text-stone-900 max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Password
                        </label>
                        <input
                            type="password"
                            placeholder="Enter password"
                            className="self-stretch px-8 py-6 text-2xl font-bold leading-4 bg-white rounded-lg border border-solid flex-[1_0_0] min-w-60 text-zinc-900 max-md:px-6 max-md:py-5 max-md:text-sm max-sm:px-5 max-sm:py-4 max-sm:text-sm"
                            name="password"
                            required
                        />
                    </div>

                    <div className="flex gap-8 items-center self-stretch">
                        <button
                            type="submit"
                            className="flex justify-center items-center gap-4 p-6 text-3xl font-bold leading-4 bg-red-900 text-white rounded-lg cursor-pointer flex-[1_0_0] max-md:p-5 max-md:text-sm max-sm:p-4 max-sm:text-sm"
                        >
                            Login
                        </button>
                    </div>

                    <div className="w-full flex justify-center">
                        <div className="text-3xl font-bold tracking-normal leading-10 text-stone-900 text-center max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Don't have an account?
                            <Link
                                to="/register"
                                className="ml-2 text-2xl font-bold text-white bg-red-900 px-4 py-2 rounded max-sm:text-sm"
                            >
                                Sign Up
                            </Link>
                        </div>
                    </div>
                </form>
            </div>
        );
    }
}

export default LoginPage;


